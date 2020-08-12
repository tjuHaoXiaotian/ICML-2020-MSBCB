import os

import numpy as np
import tensorflow as tf
from agents.common.common import scope_vars, absolute_scope_name
from simulation_env.multiuser_env import MultiUserEnv, PIDAgent
from simulation_env.multiuser_env import LearningBidAgent
from tensorflow.contrib.layers.python.layers import initializers

from replay_buffer.bid_replay_buffer import \
    PrioritizedReplayBuffer, ReplayBuffer


class BidDQN_interface(LearningBidAgent, PIDAgent):
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
            update_times_per_train=1
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

        self.scope_name = "BidDQN-model"

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
            self.replay_buffer = ReplayBuffer(self.buffer_size)

        self.margin_constant = 2

        with tf.variable_scope(self.scope_name):

            self._build_net()

            self.build_model_saver(self.scope_name)

    def _build_q_net(self, state, n_actions, variable_scope):
        with tf.variable_scope(variable_scope):
            user_id_embedding_table = tf.get_variable(
                name="user_id", shape=[self.user_num, 10], initializer=initializers.xavier_initializer(),
                trainable=True, dtype=tf.float32)
            user_id = tf.cast(state[:, 0], dtype=tf.int32)
            user_id_embeddings = tf.nn.embedding_lookup(user_id_embedding_table, ids=user_id, name="user_id_embedding")
            state = tf.concat([user_id_embeddings, state[:, 1:]], axis=1)

            n_features = state.get_shape()[1]
            fc1 = tf.layers.dense(state, units=n_features, activation=tf.nn.leaky_relu, name='fc1',
                                  kernel_initializer=initializers.xavier_initializer())

            fc2 = tf.layers.dense(fc1, units=n_features // 2, activation=tf.nn.leaky_relu, name='fc2',
                                  kernel_initializer=initializers.xavier_initializer())

            fc3 = tf.layers.dense(fc2, units=n_features // 2, activation=tf.nn.leaky_relu, name='fc3',
                                  kernel_initializer=initializers.xavier_initializer())
            q_out = tf.maximum(tf.layers.dense(fc3, units=n_actions, name='q',
                                               kernel_initializer=initializers.xavier_initializer()), 0)
            return q_out

    def _build_net(self):

        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')
        self.current_step_pctrs = tf.placeholder(tf.float32, [None], name='pctr')
        self.probability_of_not_buying = tf.placeholder(tf.float32, [None],
                                                        name='probability_of_not_buying')
        self.a = tf.placeholder(tf.int32, [None, ], name='a')
        self.done = tf.placeholder(tf.float32, [None, ], name='done')
        self.gmv_path_value = tf.placeholder(tf.float32, [None, ], name='gmv_path_value')
        self.restcost_value = tf.placeholder(tf.float32, [None, ], name='restcost_value')
        self.direct_cost_value = tf.placeholder(tf.float32, [None, ], name='restcost_value')
        self.return_value = tf.placeholder(tf.float32, [None, ], name='return')
        self.roi_thr = tf.placeholder(tf.float32, [], name="roi_thr")
        self.bid_max_ph = tf.placeholder(tf.float32, [None, ], name='bid_max')

        self.important_sampling_weight_ph = tf.placeholder(tf.float32, [None], name="important_sampling_weight")

        self.gmv_path_net = self._build_q_net(self.s, 2, variable_scope="gmv_net")
        self.cost_path_net = self._build_q_net(self.s, 2, variable_scope="cost_net")

        gmv_params = scope_vars(absolute_scope_name("gmv_net"))
        cost_params = scope_vars(absolute_scope_name("cost_net"))

        with tf.variable_scope('q_eval'):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            self.gmv_sa = tf.gather_nd(params=self.gmv_path_net, indices=a_indices)
            self.cost_sa = tf.gather_nd(params=self.cost_path_net, indices=a_indices)
            self.q_sa = self.gmv_sa - self.roi_thr * ((1 - self.done) * self.cost_sa + self.direct_cost_value)

        with tf.variable_scope('loss'):
            self._build_loss()

            self._pick_loss()

        with tf.variable_scope('train'):
            self._train_gmv_op = tf.train.AdamOptimizer(self.lr).minimize(self.gmv_loss, var_list=gmv_params)
            self._train_cost_op = tf.train.AdamOptimizer(self.lr).minimize(self.cost_loss, var_list=cost_params)
            self._train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss, var_list=gmv_params + cost_params)

        with tf.variable_scope('action'):
            profit_a_1 = (self.gmv_path_net[:, 1] - self.roi_thr * self.cost_path_net[:, 1])
            profit_a_0 = (self.gmv_path_net[:, 0] - self.roi_thr * self.cost_path_net[:, 0])
            roi_thr_times_ecpm_diff = profit_a_1 - profit_a_0
            positive_roi_thr_times_ecpm_diff = tf.maximum(roi_thr_times_ecpm_diff, 0, name="positive_diff")
            self.optimal_bid = positive_roi_thr_times_ecpm_diff / (
                    self.roi_thr * self.current_step_pctrs * self.probability_of_not_buying + 1e-10) + 0.01

        with tf.variable_scope('roi'):
            roi_min_action_1 = self.gmv_path_net[:, 1] / (self.cost_path_net[:, 1] + self.bid_max_ph + 1e-10)
            roi_action_0 = self.gmv_path_net[:, 0] / (self.cost_path_net[:, 0] + 1e-10)
            self.max_longterm_roi = tf.maximum(roi_min_action_1, roi_action_0)

    def _build_loss(self):

        if self.use_prioritized_experience_replay:

            self.gmv_montecarlo_loss = tf.reduce_mean(self.important_sampling_weight_ph *
                                                      tf.squared_difference(self.gmv_path_value, self.gmv_sa,
                                                                            name='GMV_error'))
            self.cost_montecarlo_loss = tf.reduce_mean(self.important_sampling_weight_ph *
                                                       tf.squared_difference(self.restcost_value, self.cost_sa,
                                                                             name='COST_error'))
            self.montecarlo_loss = tf.reduce_mean(self.important_sampling_weight_ph *
                                                  tf.squared_difference(self.return_value, self.q_sa,
                                                                        name='MonteCarlo_error'))
        else:

            self.gmv_montecarlo_loss = tf.reduce_mean(
                tf.squared_difference(self.gmv_path_value, self.gmv_sa, name='GMV_error'))
            self.cost_montecarlo_loss = tf.reduce_mean(
                tf.squared_difference(self.restcost_value, self.cost_sa, name='COST_error'))
            self.montecarlo_loss = tf.reduce_mean(tf.squared_difference(self.return_value, self.q_sa,
                                                                        name='MonteCarlo_error'))

        self.gmv_error = tf.abs(self.gmv_path_value - self.gmv_sa)
        self.cost_error = tf.abs(self.restcost_value - self.cost_sa)
        self.montecarlo_error = tf.abs(self.return_value - self.q_sa)

    def _pick_loss(self):
        self.gmv_loss = self.gmv_montecarlo_loss
        self.cost_loss = self.cost_montecarlo_loss
        self.loss = self.montecarlo_loss

        self.priority_values = self.gmv_error + self.cost_error + self.montecarlo_error

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
        if self.use_prioritized_experience_replay:
            buffer = self.prioritized_replay_buffer
        else:
            buffer = self.replay_buffer

        longterm_gmv, cost_restpath, return_value = 0, 0, 0
        new_gmv_cost_trajectory = other_info["gmv_and_cost"]
        for user_tuple in new_gmv_cost_trajectory[::-1]:
            state, action, [gmv, cost, reward], state_tp1, done = user_tuple
            longterm_gmv += gmv
            return_value += reward
            buffer.add(state, action, longterm_gmv, cost, cost_restpath, return_value, float(done))
            cost_restpath += cost

    def get_action(self, sess, obs, is_test=False, other_info=None):

        pctr = other_info["ctr"]
        probability_of_not_buying = other_info["probability_of_not_buying"]
        if is_test:
            bid = self.greedy_action(sess, obs, pctr, probability_of_not_buying, other_info)
        else:
            bid = self.choose_action(sess, obs, pctr, probability_of_not_buying, other_info)
        return np.clip(bid, a_min=MultiUserEnv.bid_min, a_max=MultiUserEnv.bid_max), {}

    def __epsilon_greedy__(self, sess, observation, pctr, probability_of_not_buying):
        if np.random.uniform() < self.epsilon:
            action = np.random.randint(0, self.n_actions)
            if action == 1:
                bid = MultiUserEnv.bid_max
            else:
                bid = MultiUserEnv.bid_min

        else:
            bid = self.__greedy__(sess, observation, pctr, probability_of_not_buying)
        return bid

    def __greedy__(self, sess, observation, pctr, probability_of_not_buying):
        if self.use_budget_control:
            roi_thr = self.get_roi_threshold()
        else:
            roi_thr = self.init_roi
        observations, pctrs, probabilities_of_not_buying = np.expand_dims(observation, axis=0), \
                                                           np.expand_dims(pctr, axis=0), \
                                                           np.expand_dims(probability_of_not_buying, axis=0)
        bids = sess.run(
            self.optimal_bid,
            feed_dict={
                self.s: observations,
                self.current_step_pctrs: pctrs,
                self.probability_of_not_buying: probabilities_of_not_buying,
                self.roi_thr: roi_thr,
            })
        bid = bids[0]
        return bid

    def choose_action(self, sess, observation, pctr, probability_of_not_buying, other_info):

        return self.__epsilon_greedy__(sess, observation, pctr, probability_of_not_buying)

    def greedy_action(self, sess, observation, pctr, probability_of_not_buying, other_info):
        bid = self.__greedy__(sess, observation, pctr, probability_of_not_buying)
        if self.use_budget_control:
            user_idx = other_info["user_idx"]
            request_idx = other_info["request_idx"]
            roi_threshold = self.get_roi_threshold()

            if request_idx == 0:
                observations = np.expand_dims(observation, axis=0)
                max_plongterm_roi = sess.run(
                    self.max_longterm_roi,
                    feed_dict={
                        self.s: observations,
                        self.bid_max_ph: [bid]
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
                    return 0.
        else:

            return bid

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

    def train(self, sess):
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
        if self.use_budget_control:
            roi_thr = self.get_roi_threshold()
        else:
            roi_thr = self.init_roi
        loss, montecarlo_loss, q_eval, returns = 0, 0, 0, 0
        gmv_loss, gmv_montecarlo_loss, gmv_q_eval, gmv_returns = 0, 0, 0, 0
        cost_loss, cost_montecarlo_loss, cost_q_eval, rest_costs = 0, 0, 0, 0
        for idx in range(self.update_times_per_train):
            sample_indices = self.prioritized_replay_buffer.make_index(self.batch_size)
            obses_t, actions, gmv_returns, direct_costs, rest_costs, returns, dones, weights, ranges = self.prioritized_replay_buffer.sample_index(
                sample_indices)
            _, loss, montecarlo_loss, q_eval, \
            _1, gmv_loss, gmv_montecarlo_loss, gmv_q_eval, \
            _2, cost_loss, cost_montecarlo_loss, cost_q_eval, \
            priority_values = sess.run(
                [self._train_op, self.loss, self.montecarlo_loss, self.q_sa,
                 self._train_gmv_op, self.gmv_loss, self.gmv_montecarlo_loss, self.gmv_sa,
                 self._train_cost_op, self.cost_loss, self.cost_montecarlo_loss, self.cost_sa,
                 self.priority_values],
                feed_dict={
                    self.s: obses_t,
                    self.a: actions,
                    self.gmv_path_value: gmv_returns,
                    self.direct_cost_value: direct_costs,
                    self.restcost_value: rest_costs,
                    self.roi_thr: roi_thr,
                    self.return_value: returns,
                    self.done: dones,
                    self.important_sampling_weight_ph: weights,
                })

            priorities = priority_values + 1e-6
            self.prioritized_replay_buffer.update_priorities(sample_indices, priorities)
        return loss, montecarlo_loss, np.average(q_eval), np.average(returns), \
               gmv_loss, gmv_montecarlo_loss, np.average(gmv_q_eval), np.average(gmv_returns), \
               cost_loss, cost_montecarlo_loss, np.average(cost_q_eval), np.average(rest_costs)

    def train_normal(self, sess):
        if self.use_budget_control:
            roi_thr = self.get_roi_threshold()
        else:
            roi_thr = self.init_roi

        loss, montecarlo_loss, q_eval, returns = 0, 0, 0, 0
        gmv_loss, gmv_montecarlo_loss, gmv_q_eval, gmv_returns = 0, 0, 0, 0
        cost_loss, cost_montecarlo_loss, cost_q_eval, rest_costs = 0, 0, 0, 0
        for idx in range(self.update_times_per_train):
            sample_indices = self.replay_buffer.make_index(self.batch_size)
            obses_t, actions, gmv_returns, direct_costs, rest_costs, returns, dones = self.replay_buffer.sample_index(
                sample_indices)
            _, loss, montecarlo_loss, q_eval, \
            _1, gmv_loss, gmv_montecarlo_loss, gmv_q_eval, \
            _2, cost_loss, cost_montecarlo_loss, cost_q_eval \
                = sess.run(
                [self._train_op, self.loss, self.montecarlo_loss, self.q_sa,
                 self._train_gmv_op, self.gmv_loss, self.gmv_montecarlo_loss, self.gmv_sa,
                 self._train_cost_op, self.cost_loss, self.cost_montecarlo_loss, self.cost_sa],
                feed_dict={
                    self.s: obses_t,
                    self.a: actions,
                    self.gmv_path_value: gmv_returns,
                    self.direct_cost_value: direct_costs,
                    self.restcost_value: rest_costs,
                    self.roi_thr: roi_thr,
                    self.return_value: returns,
                    self.done: dones,
                })
        return loss, montecarlo_loss, np.average(q_eval), np.average(returns), \
               gmv_loss, gmv_montecarlo_loss, np.average(gmv_q_eval), np.average(gmv_returns), \
               cost_loss, cost_montecarlo_loss, np.average(cost_q_eval), np.average(rest_costs)
