import os

import numpy as np
import numpy.random as nr
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers

from replay_buffer.replay_buffer import \
    PrioritizedReplayBuffer, ReplayBuffer
from replay_buffer.utils import add_episode
from simulation_env.multiuser_env import LearningAgent
from simulation_env.multiuser_env import PIDAgent


def scope_vars(scope, trainable_only=False):
    """
    Get variables inside a scope
    The scope can be specified as a string

    Parameters
    ----------
    scope: str or VariableScope
        scope in which the variables reside.
    trainable_only: bool
        whether or not to return only the variables that were marked as trainable.

    Returns
    -------
    vars: [tf.Variable]
        list of variables in `scope`.
    """
    return tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES if trainable_only else tf.GraphKeys.GLOBAL_VARIABLES,
        scope=scope if isinstance(scope, str) else scope.name
    )


def scope_name():
    """Returns the name of current scope as a string, e.g. deepq/q_func"""
    return tf.get_variable_scope().name


def absolute_scope_name(relative_scope_name):
    """Appends parent scope name to `relative_scope_name`"""
    return scope_name() + "/" + relative_scope_name


class OUNoise:
    """docstring for OUNoise"""

    def __init__(self, action_dimension, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * nr.randn(len(x))
        self.state = x + dx

        return self.state * 0.5


class DDPG_interface(LearningAgent, PIDAgent):
    def __init__(
            self,
            user_num,
            action_dim,
            action_bound,
            n_features,
            init_roi,
            budget,
            use_budget_control,
            use_prioritized_experience_replay,
            max_trajectory_length,
            update_times_per_train,
    ):
        PIDAgent.__init__(self, init_roi=init_roi, default_alpha=1, budget=budget, integration=2)
        self.use_budget_control = use_budget_control
        self.user_num = user_num
        self.action_bound = action_bound
        self.action_dim = action_dim
        self.n_actions = 1
        self.n_features = n_features
        self.gamma = 1.
        self.update_times_per_train = update_times_per_train

        self.lr = 0.001

        self.epsilon = 0.9
        self.epsilon_min = 0.1
        self.epsilon_dec = 0.3
        self.epsilon_dec_iter = 100

        self.replace_target_iter = 300
        self.soft_update_iter = 1
        self.softupdate = True
        self.scope_name = "DDPG-model"

        self.epoch = 0

        self.exploration_noise = OUNoise(self.action_dim)
        self.noise_weight = 1
        self.noise_descrement_per_sampling = 0.0001

        self.buffer_size = 20000 * max_trajectory_length
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

        with tf.variable_scope(self.scope_name):

            self._build_net()

            self.build_model_saver(self.scope_name)

    def _build_net(self):

        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')
        self.r_gmv = tf.placeholder(tf.float32, [None, ], name='r_gmv')
        self.r_cost = tf.placeholder(tf.float32, [None, ], name='r_cost')
        self.r = tf.placeholder(tf.float32, [None, ], name='r')
        self.roi_thr = tf.placeholder(tf.float32, [], name="roi_thr")
        self.a = tf.placeholder(tf.float32, [None, ], name='a')
        self.done = tf.placeholder(tf.float32, [None, ], name='done')
        self.gmv_return_value = tf.placeholder(tf.float32, [None, ], name='gmv_return')
        self.cost_return_value = tf.placeholder(tf.float32, [None, ], name='cost_return')
        self.return_value = tf.placeholder(tf.float32, [None, ], name='return')
        self.important_sampling_weight_ph = tf.placeholder(tf.float32, [None], name="important_sampling_weight")

        self.a_eval = self._build_action_net(self.s, variable_scope="actor_eval_net")
        self.a_target = self._build_action_net(self.s_, variable_scope="actor_target_net")
        self.gmv_critic_eval = self._build_q_net(self.s, self.a, variable_scope="gmv_critic_eval_net")
        self.gmv_critic_eval_for_loss = self._build_q_net(self.s, self.a_eval, variable_scope="gmv_critic_eval_net",
                                                          reuse=True)
        self.gmv_critic_target = self._build_q_net(self.s_, self.a_target, variable_scope="gmv_critic_target_net")

        self.cost_critic_eval = self._build_q_net(self.s, self.a, variable_scope="cost_critic_eval_net")
        self.cost_critic_eval_for_loss = self._build_q_net(self.s, self.a_eval, variable_scope="cost_critic_eval_net",
                                                           reuse=True)
        self.cost_critic_target = self._build_q_net(self.s_, self.a_target, variable_scope="cost_critic_target_net")

        self.critic_eval = self.gmv_critic_eval - self.roi_thr * self.cost_critic_eval
        self.critic_eval_for_loss = self.gmv_critic_eval_for_loss - self.roi_thr * self.cost_critic_eval_for_loss
        self.critic_target = self.gmv_critic_target - self.roi_thr * self.cost_critic_target

        ae_params = scope_vars(absolute_scope_name("actor_eval_net"))
        at_params = scope_vars(absolute_scope_name("actor_target_net"))
        gmv_ce_params = scope_vars(absolute_scope_name("gmv_critic_eval_net"))
        gmv_ct_params = scope_vars(absolute_scope_name("gmv_critic_target_net"))
        cost_ce_params = scope_vars(absolute_scope_name("cost_critic_eval_net"))
        cost_ct_params = scope_vars(absolute_scope_name("cost_critic_target_net"))
        print(ae_params)
        print(at_params)
        print(gmv_ce_params)
        print(gmv_ct_params)
        print(cost_ce_params)
        print(cost_ct_params)

        with tf.variable_scope('hard_replacement'):
            self.a_target_replace_op = tf.group([tf.assign(t, e) for t, e in zip(at_params, ae_params)])
            self.gmv_c_target_replace_op = tf.group([tf.assign(t, e) for t, e in zip(gmv_ct_params, gmv_ce_params)])
            self.cost_c_target_replace_op = tf.group([tf.assign(t, e) for t, e in zip(cost_ct_params, cost_ce_params)])

        with tf.variable_scope('soft_update'):
            self.a_update_target_q = self.__make_update_exp__(ae_params, at_params)
            self.gmv_c_update_target_q = self.__make_update_exp__(gmv_ce_params, gmv_ct_params)
            self.cost_c_update_target_q = self.__make_update_exp__(cost_ce_params, cost_ct_params)

        with tf.variable_scope('q_target'):
            self.td0_gmv_q_target = tf.stop_gradient(
                self.r_gmv + self.gamma * (1. - self.done) * self.gmv_critic_target)
            self.td0_cost_q_target = tf.stop_gradient(
                self.r_cost + self.gamma * (1. - self.done) * self.cost_critic_target)
            self.td0_q_target = tf.stop_gradient(self.r + self.gamma * (1. - self.done) * self.critic_target)

            self.montecarlo_gmv_target = self.gmv_return_value
            self.montecarlo_cost_target = self.cost_return_value
            self.montecarlo_target = self.return_value

        with tf.variable_scope('loss'):
            self._build_loss()

            self._pick_loss()

        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss,
                                                                         var_list=gmv_ce_params + cost_ce_params)
            self._train_gmv_c_op = tf.train.AdamOptimizer(self.lr).minimize(self.gmv_loss, var_list=gmv_ce_params)
            self._train_cost_c_op = tf.train.AdamOptimizer(self.lr).minimize(self.cost_loss, var_list=cost_ce_params)
            self._train_a_op = tf.train.AdamOptimizer(self.lr).minimize(self.actor_loss, var_list=ae_params)

        with tf.variable_scope('roi'):
            self.max_longterm_roi = self.gmv_critic_eval / (self.cost_critic_eval + 1e-4)

    def _pick_loss(self):

        self.has_target_net = True
        self.loss = self.td_loss
        self.gmv_loss = self.gmv_td_loss
        self.cost_loss = self.cost_td_loss
        self.actor_loss = self.a_loss
        self.priority_values = self.montecarlo_gmv_error + self.montecarlo_cost_error

    def _build_loss(self):

        if self.use_prioritized_experience_replay:

            self.gmv_td_loss = tf.reduce_mean(
                self.important_sampling_weight_ph * tf.squared_difference(self.td0_gmv_q_target, self.gmv_critic_eval,
                                                                          name='TD0_gmv_loss'))
            self.cost_td_loss = tf.reduce_mean(
                self.important_sampling_weight_ph * tf.squared_difference(self.td0_cost_q_target, self.cost_critic_eval,
                                                                          name='TD0_cost_loss'))
        else:

            self.gmv_td_loss = tf.reduce_mean(
                tf.squared_difference(self.td0_gmv_q_target, self.gmv_critic_eval, name='TD0_gmv_loss'))
            self.cost_td_loss = tf.reduce_mean(
                tf.squared_difference(self.td0_cost_q_target, self.cost_critic_eval, name='TD0_cost_loss'))
            self.td_loss = tf.reduce_mean(tf.squared_difference(self.td0_q_target, self.critic_eval, name='TD0_loss'))

        self.a_loss = - tf.reduce_mean(self.critic_eval_for_loss)

        self.gmv_montecarlo_loss = tf.reduce_mean(
            tf.squared_difference(self.montecarlo_gmv_target, self.gmv_critic_eval,
                                  name='MonteCarlo_gmv_error'))
        self.cost_montecarlo_loss = tf.reduce_mean(
            tf.squared_difference(self.montecarlo_cost_target, self.cost_critic_eval,
                                  name='MonteCarlo_cost_error'))
        self.montecarlo_loss = tf.reduce_mean(tf.squared_difference(self.montecarlo_target, self.critic_eval,
                                                                    name='MonteCarlo_error'))

        self.td0_gmv_error = tf.abs(self.td0_gmv_q_target - self.gmv_critic_eval)
        self.td0_cost_error = tf.abs(self.td0_cost_q_target - self.cost_critic_eval)
        self.td0_error = tf.abs(self.td0_q_target - self.critic_eval)

        self.montecarlo_gmv_error = tf.abs(self.montecarlo_gmv_target - self.gmv_critic_eval)
        self.montecarlo_cost_error = tf.abs(self.montecarlo_cost_target - self.cost_critic_eval)
        self.montecarlo_error = tf.abs(self.montecarlo_target - self.critic_eval)

    def _build_q_net(self, state, action, variable_scope, reuse=False):
        with tf.variable_scope(variable_scope, reuse=reuse):
            user_id_embedding_table = tf.get_variable(
                name="user_id", shape=[self.user_num, 20], initializer=initializers.xavier_initializer(),
                trainable=True, dtype=tf.float32)
            user_id = tf.cast(state[:, 0], dtype=tf.int32)
            user_id_embeddings = tf.nn.embedding_lookup(user_id_embedding_table, ids=user_id, name="user_id_embedding")
            state = tf.concat([user_id_embeddings, state[:, 1:]], axis=1)

            n_features = state.get_shape()[1]

            state = tf.concat([state, tf.expand_dims(action, axis=1, name="2d-action")], axis=1)
            fc1 = tf.layers.dense(state, units=n_features, activation=tf.nn.relu, name='fc1')
            fc2 = tf.layers.dense(fc1, units=n_features // 2, activation=tf.nn.relu, name='fc2')

            q = tf.layers.dense(fc2, units=self.action_dim, name='q')

            return q[:, 0]

    def _build_action_net(self, state, variable_scope):
        with tf.variable_scope(variable_scope):
            user_id_embedding_table = tf.get_variable(
                name="user_id", shape=[self.user_num, 20], initializer=initializers.xavier_initializer(),
                trainable=True, dtype=tf.float32)
            user_id = tf.cast(state[:, 0], dtype=tf.int32)
            user_id_embeddings = tf.nn.embedding_lookup(user_id_embedding_table, ids=user_id, name="user_id_embedding")
            state = tf.concat([user_id_embeddings, state[:, 1:]], axis=1)

            n_features = state.get_shape()[1]
            fc1 = tf.layers.dense(state, units=n_features // 2, activation=tf.nn.relu, name='fc1')
            actions = tf.layers.dense(fc1, self.action_dim, activation=tf.nn.sigmoid, name='a')
            scaled_a = tf.multiply(actions, 1, name='scaled_a')

            return scaled_a[:, 0]

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

        self.model_saver = tf.train.Saver(var_list=var_list, max_to_keep=3)

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

    def __epsilon_greedy__(self, sess, observation, roi_thr):

        if np.random.uniform() < self.epsilon:
            observation = observation[np.newaxis, :]
            actions_value = sess.run(self.a_eval, feed_dict={self.s: observation, self.roi_thr: roi_thr})

            action_noise = self.exploration_noise.noise()

            bid = actions_value + action_noise

            bid = bid[0]


        else:
            bid = self.__greedy__(sess, observation, roi_thr)

        return bid

    def __greedy__(self, sess, observation, roi_thr):

        observation = observation[np.newaxis, :]

        bid = sess.run(self.a_eval, feed_dict={self.s: observation, self.roi_thr: roi_thr})

        return bid[0]

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

        bid = self.__greedy__(sess, observation, roi_thr)
        if self.use_budget_control:
            user_idx = other_info["user_idx"]
            request_idx = other_info["request_idx"]
            roi_threshold = self.get_roi_threshold()
            if request_idx == 0:
                observations = observation[np.newaxis, :]
                max_plongterm_roi = sess.run(
                    self.max_longterm_roi,
                    feed_dict={
                        self.s: observations,
                        self.a: [bid]

                    }
                )

                if max_plongterm_roi >= roi_threshold:
                    self.explore_user(user_idx)

                    return bid
                else:

                    return 0.
            else:
                if self.is_user_selected(user_idx):

                    return bid
                else:
                    return 0
        else:

            return bid

    def get_action(self, sess, obs, is_test=False, other_info=None):
        if is_test:
            discrete_action = self.greedy_action(sess, obs, other_info)
        else:
            discrete_action = self.choose_action(sess, obs, other_info)

        other_action_info = {
            "learning_action": discrete_action
        }
        return self.action_bound * np.clip(discrete_action, 0, 1), other_action_info

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
                sess.run(self.gmv_c_update_target_q)
                sess.run(self.cost_c_update_target_q)
                sess.run(self.a_update_target_q)
        else:

            if self.epoch % self.replace_target_iter == 0:
                sess.run(self.gmv_c_update_target_q)
                sess.run(self.cost_c_update_target_q)
                sess.run(self.a_target_replace_op)

    def train(self, sess):
        if self.has_target_net:
            self.update_target(sess)

        self.epoch += 1

        if not self._is_exploration_enough(self.batch_size):
            return False, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 0, 0

        if self.use_prioritized_experience_replay:

            policy_loss, policy_entropy, loss, montecarlo_loss, q_eval, returns, \
            gmv_loss, gmv_montecarlo_loss, gmv_q_eval, gmv_returns, \
            cost_loss, cost_montecarlo_loss, cost_q_eval, cost_returns = self.train_prioritized(sess)
        else:

            policy_loss, policy_entropy, loss, montecarlo_loss, q_eval, returns, \
            gmv_loss, gmv_montecarlo_loss, gmv_q_eval, gmv_returns, \
            cost_loss, cost_montecarlo_loss, cost_q_eval, cost_returns = self.train_normal(sess)

        if self.epoch % self.epsilon_dec_iter == 0:
            self.epsilon = max(self.epsilon - self.epsilon_dec, self.epsilon_min)
            print("update epsilon:", self.epsilon)
        return True, [policy_loss, policy_entropy, loss, montecarlo_loss, q_eval, returns,
                      gmv_loss, gmv_montecarlo_loss, gmv_q_eval, gmv_returns,
                      cost_loss, cost_montecarlo_loss, cost_q_eval,
                      cost_returns], self.get_memory_returns(), self.epsilon

    def train_prioritized(self, sess):
        loss, q_eval, returns, montecarlo_loss = 0, 0, 0, 0
        for idx in range(self.update_times_per_train):
            sample_indices = self.prioritized_replay_buffer.make_index(self.batch_size)
            obs, act, rew, obs_next, done, dis_2_end, returns, weights, ranges = self.prioritized_replay_buffer.sample_index(
                sample_indices)
            _, loss, q_eval, montecarlo_loss, priority_values = sess.run(
                [self._train_c_op, self.loss, self.critic_eval, self.montecarlo_loss, self.priority_values],
                feed_dict={
                    self.s: obs,
                    self.a: act,
                    self.r: rew,
                    self.s_: obs_next,
                    self.done: done,
                    self.return_value: returns,
                    self.important_sampling_weight_ph: weights
                })
            sess.run(
                self._train_a_op,
                feed_dict={
                    self.s: obs,
                    self.a: act,
                    self.r: rew,
                    self.s_: obs_next,
                    self.done: done,
                    self.return_value: returns,
                    self.important_sampling_weight_ph: weights
                })

            priorities = priority_values + 1e-6
            self.prioritized_replay_buffer.update_priorities(sample_indices, priorities)

        return loss, montecarlo_loss, np.average(q_eval), np.average(returns)

    def train_normal(self, sess):
        policy_loss, policy_entropy = 0, 0
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
                [self._train_op, self.loss, self.montecarlo_loss, self.critic_eval,
                 self._train_gmv_c_op, self.gmv_loss, self.gmv_montecarlo_loss, self.gmv_critic_eval,
                 self._train_cost_c_op, self.cost_loss, self.cost_montecarlo_loss, self.cost_critic_eval],
                feed_dict={
                    self.s: obs,
                    self.a: act,
                    self.r_gmv: rew_gmv,
                    self.r_cost: rew_cost,
                    self.r: rew,
                    self.s_: obs_next,
                    self.done: done,
                    self.gmv_return_value: gmv_returns,
                    self.cost_return_value: cost_returns,
                    self.return_value: returns,
                    self.roi_thr: roi_thr
                })
            _, actor_loss = sess.run(
                [self._train_a_op, self.actor_loss],
                feed_dict={
                    self.roi_thr: roi_thr,
                    self.s: obs,
                    self.a: act,
                    self.r_gmv: rew_gmv,
                    self.r_cost: rew_cost,
                    self.s_: obs_next,
                    self.done: done,
                    self.gmv_return_value: gmv_returns,
                    self.cost_return_value: cost_returns,
                })

        return 0, 0, loss, montecarlo_loss, np.average(q_eval), np.average(returns), \
               gmv_loss, gmv_montecarlo_loss, np.average(gmv_q_eval), np.average(gmv_returns), \
               cost_loss, cost_montecarlo_loss, np.average(cost_q_eval), np.average(cost_returns)
