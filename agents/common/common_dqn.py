import os

import numpy as np
import tensorflow as tf
from agents.common.common import scope_vars, absolute_scope_name
from simulation_env.multiuser_env import MultiUserEnv, PIDAgent
from simulation_env.multiuser_env import LearningAgent
from tensorflow.contrib.layers.python.layers import initializers

from replay_buffer.replay_buffer import \
    PrioritizedReplayBuffer, ReplayBuffer
from replay_buffer.utils import add_episode


class DQN2Net_interface(LearningAgent, PIDAgent):

    def __init__(
            self,
            user_num,
            n_actions,
            n_features,
            init_roi,
            budget,
            use_budget_control,
            use_prioritized_experience_replay,
            max_trajectory_length,
            update_times_per_train=1,
    ):
        PIDAgent.__init__(self, init_roi=init_roi, default_alpha=1, budget=budget, integration=2)
        self.user_num = user_num
        self.use_budget_control = use_budget_control
        self.update_times_per_train = update_times_per_train
        self.n_actions = n_actions
        self.n_features = n_features
        self.gamma = 1.
        self.lr = 0.001

        self.user_based_adjust_times = 40

        self.epsilon = 0.4
        self.epsilon_min = 0.05

        self.epsilon_dec = 0.1
        self.epsilon_dec_iter = 5000 // self.user_based_adjust_times
        self.epsilon_dec_iter_min = 500 // self.user_based_adjust_times

        self.replace_target_iter = 1
        self.soft_update_iter = 1
        self.softupdate = True

        self.scope_name = "DQN-model"

        self.epoch = 0

        self.buffer_size = 1000 * max_trajectory_length

        self.batch_size = 512
        self.alpha = 0.6
        self.beta = 0.4
        self.use_prioritized_experience_replay = use_prioritized_experience_replay
        if self.use_prioritized_experience_replay:
            self.prioritized_replay_buffer = PrioritizedReplayBuffer(self.buffer_size, alpha=self.alpha,
                                                                     max_priority=20.)
        else:
            self.replay_buffer = ReplayBuffer(self.buffer_size, save_return=True)
        self.cost_replay_buffer = ReplayBuffer(self.buffer_size, save_return=True)
        self.gmv_replay_buffer = ReplayBuffer(self.buffer_size, save_return=True)

        self.margin_constant = 2

        with tf.variable_scope(self.scope_name):

            self._build_net()

            self.build_model_saver(self.scope_name)

    def _build_q_net(self, state, n_actions, variable_scope, reuse=False):
        with tf.variable_scope(variable_scope, reuse=reuse):
            user_id_embedding_table = tf.get_variable(
                name="user_id", shape=[self.user_num, 10], initializer=initializers.xavier_initializer(),
                trainable=True, dtype=tf.float32)
            user_id = tf.cast(state[:, 0], dtype=tf.int32)
            user_id_embeddings = tf.nn.embedding_lookup(user_id_embedding_table, ids=user_id, name="user_id_embedding")
            state = tf.concat([user_id_embeddings, state[:, 1:]], axis=1)

            n_features = state.get_shape()[1]

            fc1 = tf.layers.dense(state, units=n_features, activation=tf.nn.relu, name='fc1',
                                  kernel_initializer=initializers.xavier_initializer())

            fc2 = tf.layers.dense(fc1, units=n_features // 2, activation=tf.nn.relu, name='fc2',
                                  kernel_initializer=initializers.xavier_initializer())

            fc3 = tf.layers.dense(fc2, units=n_features // 2, activation=tf.nn.relu, name='fc3',
                                  kernel_initializer=initializers.xavier_initializer())
            q_out = tf.maximum(tf.layers.dense(fc3, units=n_actions, name='q',
                                               kernel_initializer=initializers.xavier_initializer()), 0)
            return q_out

    def _build_net(self):

        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')
        self.r_gmv = tf.placeholder(tf.float32, [None, ], name='r_gmv')
        self.r_cost = tf.placeholder(tf.float32, [None, ], name='r_cost')
        self.roi_thr = tf.placeholder(tf.float32, [], name="roi_thr")
        self.r = tf.placeholder(tf.float32, [None, ], name='r')
        self.a = tf.placeholder(tf.int32, [None, ], name='a')
        self.done = tf.placeholder(tf.float32, [None, ], name='done')
        self.return_gmv_value = tf.placeholder(tf.float32, [None, ], name='return_gmv')
        self.return_cost_value = tf.placeholder(tf.float32, [None, ], name='return_cost')
        self.return_value = tf.placeholder(tf.float32, [None, ], name='return')
        self.important_sampling_weight_ph = tf.placeholder(tf.float32, [None], name="important_sampling_weight")

        self.q_eval_gmv = self._build_q_net(self.s, self.n_actions, variable_scope="eval_gmv_net")
        self.q_next_gmv = self._build_q_net(self.s_, self.n_actions, variable_scope="target_gmv_net")
        self.q_eval_cost = self._build_q_net(self.s, self.n_actions, variable_scope="eval_cost_net")
        self.q_next_cost = self._build_q_net(self.s_, self.n_actions, variable_scope="target_cost_net")
        self.q_eval = self.q_eval_gmv - self.roi_thr * self.q_eval_cost
        self.q_next = self.q_next_gmv - self.roi_thr * self.q_next_cost

        t_gmv_params = scope_vars(absolute_scope_name("target_gmv_net"))
        e_gmv_params = scope_vars(absolute_scope_name("eval_gmv_net"))
        t_cost_params = scope_vars(absolute_scope_name("target_cost_net"))
        e_cost_params = scope_vars(absolute_scope_name("eval_cost_net"))

        with tf.variable_scope('hard_replacement'):
            self.target_gmv_replace_op = tf.group([tf.assign(t, e) for t, e in zip(t_gmv_params, e_gmv_params)])
            self.target_cost_replace_op = tf.group([tf.assign(t, e) for t, e in zip(t_cost_params, e_cost_params)])

        with tf.variable_scope('soft_update'):
            self.update_gmv_target_q = self.__make_update_exp__(e_gmv_params, t_gmv_params)
            self.update_cost_target_q = self.__make_update_exp__(e_cost_params, t_cost_params)

        with tf.variable_scope('q_target'):
            greedy_action_s_ = tf.argmax(self.q_next, axis=-1, name="td0_argmax_action", output_type=tf.int32)
            greedy_a_indices = tf.stack(
                [tf.range(tf.cast(tf.shape(self.a)[0], dtype=tf.int32), dtype=tf.int32), greedy_action_s_],
                axis=1)
            target_q_gmv_sa = tf.gather_nd(params=self.q_next_gmv, indices=greedy_a_indices)
            target_q_cost_sa = tf.gather_nd(params=self.q_next_cost, indices=greedy_a_indices)
            target_q_sa = tf.gather_nd(params=self.q_next, indices=greedy_a_indices)
            self.td0_q_gmv_target = tf.stop_gradient(self.r_gmv + self.gamma * (1. - self.done) * target_q_gmv_sa)
            self.td0_q_cost_target = tf.stop_gradient(self.r_cost + self.gamma * (1. - self.done) * target_q_cost_sa)
            self.td0_q_target = tf.stop_gradient(self.r + self.gamma * (1. - self.done) * target_q_sa)

            target_action = tf.argmax(self.q_eval, axis=-1, name="doubeldqn_argmax_action",
                                      output_type=tf.int32)
            target_a_indices = tf.stack(
                [tf.range(tf.cast(tf.shape(self.a)[0], dtype=tf.int32), dtype=tf.int32), target_action],
                axis=1)
            ddqn_target_q_gmv_sa = tf.gather_nd(params=self.q_next_gmv, indices=target_a_indices)
            ddqn_target_q_cost_sa = tf.gather_nd(params=self.q_next_cost, indices=target_a_indices)
            ddqn_target_q_sa = tf.gather_nd(params=self.q_next, indices=target_a_indices)
            self.double_dqn_gmv_target = tf.stop_gradient(
                self.r_gmv + self.gamma * (1. - self.done) * ddqn_target_q_gmv_sa)
            self.double_dqn_cost_target = tf.stop_gradient(
                self.r_cost + self.gamma * (1. - self.done) * ddqn_target_q_cost_sa)
            self.double_dqn_target = tf.stop_gradient(self.r + self.gamma * (1. - self.done) * ddqn_target_q_sa)

            self.montecarlo_gmv_target = self.return_gmv_value
            self.montecarlo_cost_target = self.return_cost_value
            self.montecarlo_target = self.return_value

        with tf.variable_scope('q_eval'):
            a_indices = tf.stack([tf.range(tf.cast(tf.shape(self.a)[0], dtype=tf.int32), dtype=tf.int32), self.a],
                                 axis=1)
            self.q_eval_gmv_wrt_a = tf.gather_nd(params=self.q_eval_gmv, indices=a_indices)
            self.q_eval_cost_wrt_a = tf.gather_nd(params=self.q_eval_cost, indices=a_indices)
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)

        with tf.variable_scope('loss'):
            self._build_loss()

            self._pick_loss()

        with tf.variable_scope('train'):
            self._train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss, var_list=e_gmv_params + e_cost_params)
            self._train_gmv_op = tf.train.AdamOptimizer(self.lr).minimize(self.gmv_loss, var_list=e_gmv_params)
            self._train_cost_op = tf.train.AdamOptimizer(self.lr).minimize(self.cost_loss, var_list=e_cost_params)

        with tf.variable_scope('roi'):
            greedy_action_indices = tf.stack(
                [tf.range(tf.cast(tf.shape(self.a)[0], dtype=tf.int32), dtype=tf.int32), self.a], axis=1)
            self.plongterm_roi = tf.gather_nd(params=self.q_eval_gmv, indices=greedy_action_indices) / (
                    tf.gather_nd(params=self.q_eval_cost, indices=greedy_action_indices) + 1e-6)

    def _pick_loss(self):
        self.has_target_net = True
        self.gmv_loss = self.gmv_double_dqn_loss
        self.cost_loss = self.cost_double_dqn_loss
        self.loss = self.double_dqn_loss
        self.priority_values = self.gmv_doubel_dqn_error + self.cost_doubel_dqn_error + self.doubel_dqn_error

    def _build_loss(self):

        if self.use_prioritized_experience_replay:

            self.gmv_dqn_loss = tf.reduce_mean(
                self.important_sampling_weight_ph * tf.squared_difference(self.td0_q_gmv_target, self.q_eval_gmv_wrt_a,
                                                                          name='TD0_gmv_loss'))
            self.cost_dqn_loss = tf.reduce_mean(
                self.important_sampling_weight_ph * tf.squared_difference(self.td0_q_cost_target,
                                                                          self.q_eval_cost_wrt_a,
                                                                          name='TD0_cost_loss'))
            self.dqn_loss = tf.reduce_mean(
                self.important_sampling_weight_ph * tf.squared_difference(self.td0_q_target, self.q_eval_wrt_a,
                                                                          name='TD0_loss'))

            self.gmv_double_dqn_loss = tf.reduce_mean(
                self.important_sampling_weight_ph * tf.squared_difference(self.double_dqn_gmv_target,
                                                                          self.q_eval_gmv_wrt_a,
                                                                          name='Double_DQN_gmv_loss'))
            self.cost_double_dqn_loss = tf.reduce_mean(
                self.important_sampling_weight_ph * tf.squared_difference(self.double_dqn_cost_target,
                                                                          self.q_eval_cost_wrt_a,
                                                                          name='Double_DQN_cost_loss'))
            self.double_dqn_loss = tf.reduce_mean(
                self.important_sampling_weight_ph * tf.squared_difference(self.double_dqn_target, self.q_eval_wrt_a,
                                                                          name='Double_DQN_error'))

            self.gmv_montecarlo_loss = tf.reduce_mean(self.important_sampling_weight_ph *
                                                      tf.squared_difference(self.montecarlo_gmv_target,
                                                                            self.q_eval_gmv_wrt_a,
                                                                            name='GMV_error'))
            self.cost_montecarlo_loss = tf.reduce_mean(self.important_sampling_weight_ph *
                                                       tf.squared_difference(self.montecarlo_cost_target,
                                                                             self.q_eval_cost_wrt_a,
                                                                             name='COST_error'))
            self.montecarlo_loss = tf.reduce_mean(self.important_sampling_weight_ph *
                                                  tf.squared_difference(self.montecarlo_target, self.q_eval_wrt_a,
                                                                        name='MonteCarlo_error'))

        else:

            self.gmv_dqn_loss = tf.reduce_mean(
                tf.squared_difference(self.td0_q_gmv_target, self.q_eval_gmv_wrt_a, name='TD0_gmv_loss'))
            self.cost_dqn_loss = tf.reduce_mean(
                tf.squared_difference(self.td0_q_cost_target, self.q_eval_cost_wrt_a, name='TD0_cost_loss'))
            self.dqn_loss = tf.reduce_mean(tf.squared_difference(self.td0_q_target, self.q_eval_wrt_a, name='TD0_loss'))

            self.gmv_double_dqn_loss = tf.reduce_mean(
                tf.squared_difference(self.double_dqn_gmv_target, self.q_eval_gmv_wrt_a,
                                      name='Double_DQN_gmv_loss'))
            self.cost_double_dqn_loss = tf.reduce_mean(
                tf.squared_difference(self.double_dqn_cost_target, self.q_eval_cost_wrt_a,
                                      name='Double_DQN_cost_loss'))
            self.double_dqn_loss = tf.reduce_mean(tf.squared_difference(self.double_dqn_target, self.q_eval_wrt_a,
                                                                        name='Double_DQN_error'))

            self.gmv_montecarlo_loss = tf.reduce_mean(
                tf.squared_difference(self.montecarlo_gmv_target, self.q_eval_gmv_wrt_a,
                                      name='MonteCarlo_gmv_loss'))
            self.cost_montecarlo_loss = tf.reduce_mean(
                tf.squared_difference(self.montecarlo_cost_target, self.q_eval_cost_wrt_a,
                                      name='MonteCarlo_cost_loss'))
            self.montecarlo_loss = tf.reduce_mean(tf.squared_difference(self.montecarlo_target, self.q_eval_wrt_a,
                                                                        name='MonteCarlo_error'))

        self.gmv_td0_error = tf.abs(self.td0_q_gmv_target - self.q_eval_gmv_wrt_a)
        self.cost_td0_error = tf.abs(self.td0_q_cost_target - self.q_eval_cost_wrt_a)
        self.td0_error = tf.abs(self.td0_q_target - self.q_eval_wrt_a)

        self.gmv_doubel_dqn_error = tf.abs(self.double_dqn_gmv_target - self.q_eval_gmv_wrt_a)
        self.cost_doubel_dqn_error = tf.abs(self.double_dqn_cost_target - self.q_eval_cost_wrt_a)
        self.doubel_dqn_error = tf.abs(self.double_dqn_target - self.q_eval_wrt_a)

        self.gmv_montecarlo_error = tf.abs(self.montecarlo_gmv_target - self.q_eval_gmv_wrt_a)
        self.cost_montecarlo_error = tf.abs(self.montecarlo_cost_target - self.q_eval_cost_wrt_a)
        self.montecarlo_error = tf.abs(self.montecarlo_target - self.q_eval_wrt_a)

    def __make_update_exp__(self, vals, target_vals):
        polyak = 1.0 - 1e-2
        expression = []
        for var, var_target in zip(sorted(vals, key=lambda v: v.name), sorted(target_vals, key=lambda v: v.name)):
            expression.append(var_target.assign(polyak * var_target + (1.0 - polyak) * var))
        expression = tf.group(*expression)
        return expression

    def __make_hardreplace_exp__(self, vals, target_vals):
        expression = []
        for var, var_target in zip(sorted(vals, key=lambda v: v.name), sorted(target_vals, key=lambda v: v.name)):
            expression.append(var_target.assign(var))

        expression = tf.group(*expression)
        return expression

    def build_model_saver(self, var_scope):
        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=var_scope)

        self.model_saver = tf.train.Saver(var_list=var_list, max_to_keep=1)

    def save(self, sess, path, step):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        self.model_saver.save(sess, save_path=path, global_step=step)

    def restore(self, sess, path):
        self.model_saver.restore(sess, save_path=path)
        print('%s model reloaded from %s' % (self.scope_name, path))

    def experience(self, new_trajectory, other_info=None):
        new_trajectory_gmv = other_info["gmv"]
        new_trajectory_cost = other_info["cost"]
        if self.use_prioritized_experience_replay:
            add_episode(self.prioritized_replay_buffer, new_trajectory, gamma=self.gamma)
        else:
            add_episode(self.replay_buffer, new_trajectory, gamma=self.gamma)

        add_episode(self.gmv_replay_buffer, new_trajectory_gmv, gamma=self.gamma)
        add_episode(self.cost_replay_buffer, new_trajectory_cost, gamma=self.gamma)

    def get_action(self, sess, obs, is_test=False, other_info=None):

        if is_test:
            discrete_action = self.greedy_action(sess, obs, other_info)
        else:
            discrete_action = self.choose_action(sess, obs, other_info)
        bid_max = MultiUserEnv.bid_max
        bid_min = MultiUserEnv.bid_min
        other_action_info = {
            "learning_action": discrete_action
        }
        return bid_min + (bid_max - bid_min) / (self.n_actions - 1) * discrete_action, other_action_info

    def __greedy__(self, sess, observation, roi_thr):
        observations = observation[np.newaxis, :]
        actions_value = sess.run(self.q_eval, feed_dict={self.s: observations, self.roi_thr: roi_thr})
        greedy_action = np.argmax(actions_value, axis=1)[0]
        return greedy_action

    def __epsilon_greedy__(self, sess, observation, roi_thr):
        if np.random.uniform() < self.epsilon:
            action = np.random.randint(0, self.n_actions)
        else:
            action = self.__greedy__(sess, observation, roi_thr)
        return action

    def choose_action(self, sess, observation, other_info):
        if self.use_budget_control:
            roi_thr = self.get_roi_threshold()
        else:
            roi_thr = self.init_roi

        return self.__epsilon_greedy__(sess, observation, roi_thr)

    def greedy_action(self, sess, observation, other_info):
        if self.use_budget_control:
            roi_thr = self.get_roi_threshold()
        else:
            roi_thr = self.init_roi

        greedy_action = self.__greedy__(sess, observation, roi_thr)
        if self.use_budget_control:
            user_idx = other_info["user_idx"]
            request_idx = other_info["request_idx"]
            roi_threshold = self.get_roi_threshold()
            if request_idx == 0:
                observations = np.expand_dims(observation, axis=0)
                max_plongterm_roi = sess.run(
                    self.plongterm_roi,
                    feed_dict={
                        self.s: observations,
                        self.a: [greedy_action],
                    }
                )
                if max_plongterm_roi >= roi_threshold:
                    self.explore_user(user_idx)
                    return greedy_action
                else:
                    return 0
            else:
                if self.is_user_selected(user_idx):
                    return greedy_action
                else:
                    return 0
        else:
            return greedy_action

    def get_memory_returns(self):
        if self.use_prioritized_experience_replay:
            return self.prioritized_replay_buffer.current_mean_return
        else:
            return self.replay_buffer.current_mean_return

    def _is_exploration_enough(self, min_pool_size):
        if self.use_prioritized_experience_replay:
            return len(self.prioritized_replay_buffer) >= min_pool_size
        else:
            return len(self.replay_buffer) >= min_pool_size

    def update_target(self, sess):
        if self.softupdate:

            if self.epoch % self.soft_update_iter == 0:
                sess.run(self.update_gmv_target_q)
                sess.run(self.update_cost_target_q)
        else:

            if self.epoch % self.replace_target_iter == 0:
                sess.run(self.target_gmv_replace_op)
                sess.run(self.target_cost_replace_op)

    def train(self, sess):
        if self.has_target_net:
            self.update_target(sess)

        self.epoch += 1

        if not self._is_exploration_enough(self.batch_size):
            return False, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 0, 0

        if self.use_prioritized_experience_replay:

            loss, montecarlo_loss, q_eval, returns, \
            gmv_loss, gmv_montecarlo_loss, gmv_q_eval, gmv_returns, \
            cost_loss, cost_montecarlo_loss, cost_q_eval, cost_returns = self.train_prioritized(sess)
        else:

            loss, montecarlo_loss, q_eval, returns, \
            gmv_loss, gmv_montecarlo_loss, gmv_q_eval, gmv_returns, \
            cost_loss, cost_montecarlo_loss, cost_q_eval, cost_returns = self.train_normal(sess)

        if self.epoch % self.epsilon_dec_iter == 0:
            self.epsilon = max(self.epsilon - self.epsilon_dec, self.epsilon_min)
            self.epsilon_dec_iter //= 1.5
            self.epsilon_dec_iter = max(self.epsilon_dec_iter, self.epsilon_dec_iter_min)
            print("update epsilon:", self.epsilon)
        return True, [0, 0, loss, montecarlo_loss, q_eval, returns,
                      gmv_loss, gmv_montecarlo_loss, gmv_q_eval, gmv_returns,
                      cost_loss, cost_montecarlo_loss, cost_q_eval,
                      cost_returns], self.get_memory_returns(), self.epsilon

    def train_prioritized(self, sess):
        loss, montecarlo_loss, q_eval, returns = 0, 0, 0, 0
        gmv_loss, gmv_montecarlo_loss, gmv_q_eval, gmv_returns = 0, 0, 0, 0
        cost_loss, cost_montecarlo_loss, cost_q_eval, cost_returns = 0, 0, 0, 0
        if self.use_budget_control:
            roi_thr = self.get_roi_threshold()
        else:
            roi_thr = self.init_roi
        for idx in range(self.update_times_per_train):
            sample_indices = self.prioritized_replay_buffer.make_index(self.batch_size)
            obs, act, rew, obs_next, done, dis_2_end, returns, weights, ranges = self.prioritized_replay_buffer.sample_index(
                sample_indices)
            obs, act, rew_gmv, obs_next, done, dis_2_end, gmv_returns, weights, ranges = self.gmv_replay_buffer.sample_index(
                sample_indices)
            obs, act, rew_cost, obs_next, done, dis_2_end, cost_returns = self.cost_replay_buffer.sample_index(
                sample_indices)
            _, loss, montecarlo_loss, q_eval, \
            _1, gmv_loss, gmv_montecarlo_loss, gmv_q_eval, \
            _2, cost_loss, cost_montecarlo_loss, cost_q_eval, \
            priority_values = sess.run(
                [self._train_op, self.loss, self.montecarlo_loss, self.q_eval_wrt_a,
                 self._train_gmv_op, self.gmv_loss, self.gmv_montecarlo_loss, self.q_eval_gmv_wrt_a,
                 self._train_cost_op, self.cost_loss, self.cost_montecarlo_loss, self.q_eval_cost_wrt_a,
                 self.priority_values],
                feed_dict={
                    self.s: obs,
                    self.a: act,
                    self.r_gmv: rew_gmv,
                    self.r_cost: rew_cost,
                    self.r: rew,
                    self.s_: obs_next,
                    self.done: done,
                    self.return_gmv_value: gmv_returns,
                    self.return_cost_value: cost_returns,
                    self.return_value: returns,
                    self.important_sampling_weight_ph: weights,
                    self.roi_thr: roi_thr
                })

            priorities = priority_values + 1e-6
            self.prioritized_replay_buffer.update_priorities(sample_indices, priorities)
        return loss, montecarlo_loss, np.average(q_eval), np.average(returns), \
               gmv_loss, gmv_montecarlo_loss, np.average(gmv_q_eval), np.average(gmv_returns), \
               cost_loss, cost_montecarlo_loss, np.average(cost_q_eval), np.average(cost_returns)

    def train_normal(self, sess):
        loss, montecarlo_loss, q_eval, returns = 0, 0, 0, 0
        gmv_loss, gmv_montecarlo_loss, gmv_q_eval, gmv_returns = 0, 0, 0, 0
        cost_loss, cost_montecarlo_loss, cost_q_eval, cost_returns = 0, 0, 0, 0
        if self.use_budget_control:
            roi_thr = self.get_roi_threshold()
        else:
            roi_thr = self.init_roi
        for idx in range(self.update_times_per_train):
            sample_indices = self.replay_buffer.make_index(self.batch_size)

            obs, act, rew, obs_next, done, dis_2_end, returns = self.replay_buffer.sample_index(
                sample_indices)
            obs, act, rew_gmv, obs_next, done, dis_2_end, gmv_returns = self.gmv_replay_buffer.sample_index(
                sample_indices)
            obs, act, rew_cost, obs_next, done, dis_2_end, cost_returns = self.cost_replay_buffer.sample_index(
                sample_indices)

            _, loss, montecarlo_loss, q_eval, \
            _1, gmv_loss, gmv_montecarlo_loss, gmv_q_eval, \
            _2, cost_loss, cost_montecarlo_loss, cost_q_eval \
                = sess.run(
                [self._train_op, self.loss, self.montecarlo_loss, self.q_eval_wrt_a,

                 self._train_gmv_op, self.gmv_loss, self.gmv_montecarlo_loss, self.q_eval_gmv_wrt_a,
                 self._train_cost_op, self.cost_loss, self.cost_montecarlo_loss, self.q_eval_cost_wrt_a],
                feed_dict={
                    self.s: obs,
                    self.a: act,
                    self.r_gmv: rew_gmv,
                    self.r_cost: rew_cost,
                    self.r: rew,
                    self.s_: obs_next,
                    self.done: done,
                    self.return_gmv_value: gmv_returns,
                    self.return_cost_value: cost_returns,
                    self.return_value: returns,
                    self.roi_thr: roi_thr
                })
        return loss, montecarlo_loss, np.average(q_eval), np.average(returns), \
               gmv_loss, gmv_montecarlo_loss, np.average(gmv_q_eval), np.average(gmv_returns), \
               cost_loss, cost_montecarlo_loss, np.average(cost_q_eval), np.average(cost_returns)
