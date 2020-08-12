import os
import random

import numpy as np
import tensorflow as tf
from simulation_env.multiuser_env import MultiUserEnv, PIDAgent
from simulation_env.multiuser_env import LearningAgent
from tensorflow.contrib.layers.python.layers import initializers

from replay_buffer.replay_buffer import \
    PrioritizedReplayBuffer, ReplayBuffer
from replay_buffer.utils import add_episode


def set_seed(seed):
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


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


def reduce_var(x, axis=None, keepdims=False):
    """Variance of a tensor, alongside the specified axis.


        x: A tensor or variable.
        axis: An integer, the axis to compute the variance.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.


        A tensor with the variance of elements of `x`.
    """
    m = tf.reduce_mean(x, axis=axis, keep_dims=True)
    devs_squared = tf.square(x - m)
    return tf.reduce_mean(devs_squared, axis=axis, keep_dims=keepdims)


def reduce_std(x, axis=None, keepdims=False):
    """Standard deviation of a tensor, alongside the specified axis.


        x: A tensor or variable.
        axis: An integer, the axis to compute the standard deviation.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.


        A tensor with the standard deviation of elements of `x`.
    """
    return tf.sqrt(reduce_var(x, axis=axis, keepdims=keepdims))


class PPO_interface(LearningAgent, PIDAgent):

    def __init__(self,
                 user_num,
                 action_dim,
                 n_features,
                 init_roi,
                 budget,
                 use_budget_control,
                 use_prioritized_experience_replay,
                 max_trajectory_length,
                 update_times_per_train
                 ):
        PIDAgent.__init__(self, init_roi=init_roi, default_alpha=1, budget=budget, integration=1)
        self.user_num = user_num
        self.use_budget_control = use_budget_control
        self.action_dim = action_dim
        self.n_actions = 11
        self.n_features = n_features
        self.lr = 0.001
        self.update_times_per_train = update_times_per_train

        self.epsilon = 0.5
        self.epsilon_min = 0.01
        self.epsilon_dec = 0.2
        self.epsilon_dec_iter = 100

        self.epsilon_clip = 0.2
        self.replace_target_iter = 1
        self.soft_update_iter = 1
        self.softupdate = False
        self.scope_name = "PPO-model"

        self.epoch = 0
        self.lam = 0.5

        self.update_step = 1
        self.kl_target = 0.01
        self.gamma = 1.
        self.method = 'clip'

        self.policy_logvar = 1e-7

        self.decay_rate = 0.9
        self.decay_steps = 5000

        self.global_ = tf.Variable(tf.constant(0))

        self.buffer_size = 1000 * max_trajectory_length
        self.batch_size = 500
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

        self.r_gmv = tf.placeholder(tf.float32, [None, ], name='r_gmv')
        self.r_cost = tf.placeholder(tf.float32, [None, ], name='r_cost')
        self.roi_thr = tf.placeholder(tf.float32, [], name="roi_thr")
        self.r = tf.placeholder(tf.float32, [None, ], name='r')
        self.a = tf.placeholder(tf.int32, [None, ], name='a')
        self.adv = tf.placeholder(tf.float32, [None, ], name='advantage')
        self.done = tf.placeholder(tf.float32, [None, ], name='done')
        self.gmv_return_value = tf.placeholder(tf.float32, [None, ], name='gmv_return')
        self.cost_return_value = tf.placeholder(tf.float32, [None, ], name='cost_return')
        self.return_value = tf.placeholder(tf.float32, [None, ], name='return')
        self.important_sampling_weight_ph = tf.placeholder(tf.float32, [None], name="important_sampling_weight")

        self.a_eval = self._build_action_net(self.s, variable_scope="actor_eval_net")
        self.a_target = self._build_action_net(self.s, variable_scope="actor_target_net")
        self.critic_gmv = self._build_q_net(self.s, variable_scope="critic_eval_gmv_net")
        self.critic_cost = self._build_q_net(self.s, variable_scope="critic_eval_cost_net")
        self.critic = self.critic_gmv - self.roi_thr * self.critic_cost

        ae_params = scope_vars(absolute_scope_name("actor_eval_net"))
        at_params = scope_vars(absolute_scope_name("actor_target_net"))

        print(ae_params)
        print(at_params)

        with tf.variable_scope('hard_replacement'):
            self.a_target_replace_op = tf.group([tf.assign(t, e) for t, e in zip(at_params, ae_params)])

        self._build_loss()

        self._pick_loss()

        with tf.variable_scope('train'):
            self.gmv_ctrain_op = tf.train.AdamOptimizer(self.lr).minimize(self.gmv_loss)
            self.cost_ctrain_op = tf.train.AdamOptimizer(self.lr).minimize(self.cost_loss)
            self.ctrain_op = tf.train.AdamOptimizer(self.lr).minimize(self.critic_loss)
            self.atrain_op = tf.train.AdamOptimizer(self.lr).minimize(self.actor_loss)

        with tf.variable_scope('roi'):
            self.max_longterm_roi = self.critic_gmv / (self.critic_cost + 1e-4)

    def _pick_loss(self):
        self.has_target_net = True
        self.critic_loss = self.closs

        self.gmv_loss = self.gmv_closs
        self.cost_loss = self.cost_closs
        self.actor_loss = self.aloss

    def _build_loss(self):
        with tf.variable_scope('critic'):
            self.gmv_c_loss = self.gmv_return_value - self.critic_gmv
            self.cost_c_loss = self.cost_return_value - self.critic_cost
            self.c_loss = self.return_value - self.critic

            self.gmv_closs = tf.reduce_mean(tf.square(self.gmv_c_loss))
            self.cost_closs = tf.reduce_mean(tf.square(self.cost_c_loss))
            self.closs = tf.reduce_mean(tf.square(self.c_loss))

            self.advantage = self.return_value - self.critic

        with tf.variable_scope('surrogate'):

            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            pi_prob = tf.gather_nd(params=self.a_eval, indices=a_indices)
            oldpi_prob = tf.gather_nd(params=self.a_target, indices=a_indices)
            ratio = pi_prob / (oldpi_prob + 1e-8)
            surr = ratio * self.adv
            if self.method == 'kl_pen':

                kl = tf.distributions.kl_divergence(self.a_target, self.a_eval)
                self.kl_mean = tf.reduce_mean(kl)
                self.aloss = -(tf.reduce_mean(surr - self.lam * kl))
            else:
                self.aloss = -tf.reduce_mean(tf.minimum(
                    surr,
                    tf.clip_by_value(ratio, 1. - self.epsilon_clip, 1. + self.epsilon_clip) * self.adv))

    def update_target(self, sess):
        if self.epoch % self.replace_target_iter == 0:
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

    def _build_action_net(self, state, variable_scope):
        with tf.variable_scope(variable_scope):
            user_id_embedding_table = tf.get_variable(
                name="user_id", shape=[self.user_num, 20], initializer=initializers.xavier_initializer(),
                trainable=True, dtype=tf.float32)
            user_id = tf.cast(state[:, 0], dtype=tf.int32)
            user_id_embeddings = tf.nn.embedding_lookup(user_id_embedding_table, ids=user_id, name="user_id_embedding")
            state = tf.concat([user_id_embeddings, state[:, 1:]], axis=1)

            n_features = state.get_shape()[1]

            l1 = tf.layers.dense(state, n_features // 2, tf.nn.relu)
            l2 = tf.layers.dense(l1, n_features // 4, tf.nn.relu)
            a_prob = tf.layers.dense(l2, self.n_actions, tf.nn.softmax)
        return a_prob

    def _build_q_net(self, state, variable_scope, reuse=False):

        with tf.variable_scope(variable_scope, reuse=reuse):
            user_id_embedding_table = tf.get_variable(
                name="user_id", shape=[self.user_num, 20], initializer=initializers.xavier_initializer(),
                trainable=True, dtype=tf.float32)
            user_id = tf.cast(state[:, 0], dtype=tf.int32)
            user_id_embeddings = tf.nn.embedding_lookup(user_id_embedding_table, ids=user_id, name="user_id_embedding")
            state = tf.concat([user_id_embeddings, state[:, 1:]], axis=1)

            n_features = state.get_shape()[1]

            l1 = tf.layers.dense(state, n_features // 2, tf.nn.relu)
            l2 = tf.layers.dense(l1, n_features // 4, tf.nn.relu)
            v = tf.layers.dense(l2, 1)
        return v[:, 0]

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

            sample_indices = self.replay_buffer.make_latest_index(self.batch_size)

            obs, act, rew, obs_next, done, dis_2_end, returns = self.replay_buffer.sample_index(
                sample_indices)
            obs, act, rew_gmv, obs_next, done, dis_2_end, gmv_returns = self.gmv_replay_buffer.sample_index(
                sample_indices)
            obs, act, rew_cost, obs_next, done, dis_2_end, cost_returns = self.cost_replay_buffer.sample_index(
                sample_indices)

            adv = sess.run(self.advantage, {self.s: obs, self.return_value: returns, self.roi_thr: roi_thr})

            ret = sess.run(self.return_value, {self.s: obs, self.return_value: returns, self.roi_thr: roi_thr})

            criti = sess.run(self.critic_cost, {self.s: obs, self.return_value: returns, self.roi_thr: roi_thr})

            [sess.run([self.ctrain_op, self.gmv_ctrain_op, self.cost_ctrain_op], feed_dict={
                self.adv: adv,
                self.s: obs,
                self.a: act,
                self.r_gmv: rew_gmv,
                self.r_cost: rew_cost,
                self.r: rew,
                self.done: done,
                self.gmv_return_value: gmv_returns,
                self.cost_return_value: cost_returns,
                self.return_value: returns,
                self.roi_thr: roi_thr}) for _ in range(self.update_step)]

            if self.method == 'kl_pen':
                for _ in range(self.update_step):
                    _, kl, loss, gmv_eval, cost_eval = sess.run(
                        [self.atrain_op, self.kl_mean, self.closs, self.critic_gmv, self.critic_cost],
                        feed_dict={
                            self.adv: adv,
                            self.s: obs,
                            self.a: act,
                            self.r_gmv: rew_gmv,
                            self.r_cost: rew_cost,
                            self.r: rew,

                            self.done: done,
                            self.gmv_return_value: gmv_returns,
                            self.cost_return_value: cost_returns,
                            self.return_value: returns,
                            self.roi_thr: roi_thr})
                    if kl > 4 * self.kl_target:
                        break
                if kl < self.kl_target / 1.5:
                    self.lam /= 2
                elif kl > self.kl_target * 1.5:
                    self.lam *= 2
                self.lam = np.clip(self.lam, 1e-4, 10)
            else:

                for _ in range(self.update_step):
                    _, loss, q_eval, gmv_loss, gmv_q_eval, cost_loss, cost_q_eval \
                        = sess.run(
                        [self.atrain_op, self.closs, self.critic,
                         self.gmv_loss, self.critic_gmv,
                         self.cost_loss, self.critic_cost],
                        feed_dict={
                            self.adv: adv,
                            self.s: obs,
                            self.a: act,
                            self.r_gmv: rew_gmv,
                            self.r_cost: rew_cost,
                            self.r: rew,

                            self.done: done,
                            self.gmv_return_value: gmv_returns,
                            self.cost_return_value: cost_returns,
                            self.return_value: returns,
                            self.roi_thr: roi_thr
                        })

        return policy_loss, policy_entropy, loss, montecarlo_loss, np.average(q_eval), np.average(returns), \
               gmv_loss, gmv_montecarlo_loss, np.average(gmv_q_eval), np.average(gmv_returns), \
               cost_loss, cost_montecarlo_loss, np.average(cost_q_eval), np.average(cost_returns)

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

            s = observation[np.newaxis, :]

            prob_weights = sess.run(self.a_eval, feed_dict={self.s: s})
            a = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())

            bid = a

        else:

            bid = self.__greedy__(sess, observation, roi_thr)

        return bid

    def __greedy__(self, sess, observation, roi_thr):

        s = observation[np.newaxis, :]

        prob_weights = sess.run(self.a_eval, feed_dict={self.s: s})
        a = np.argmax(prob_weights, axis=1)[0]
        bid = a

        return bid

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
        bid_max = MultiUserEnv.bid_max
        bid_min = MultiUserEnv.bid_min

        other_action_info = {
            "learning_action": discrete_action
        }
        return bid_min + (bid_max - bid_min) / (self.n_actions - 1) * discrete_action, other_action_info

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
