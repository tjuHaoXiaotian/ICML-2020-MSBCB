import os
import random

import numpy as np
import tensorflow as tf
from simulation_env.multiuser_env import LearningAgent

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


class DQN_interface(LearningAgent):
    def __init__(
            self,
            n_actions=11,
            n_features=29,
            use_prioritized_experience_replay=True,
            max_trajectory_length=20,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.gamma = 1.

        self.lr = 0.001
        self.epsilon = 0.5
        self.epsilon_min = 0
        self.epsilon_dec = 0.1
        self.epsilon_dec_iter = 1000
        self.replace_target_iter = 100
        self.soft_update_iter = 1
        self.softupdate = False
        self.scope_name = "DQN-model"

        self.epoch = 0

        self.buffer_size = 5000 * max_trajectory_length
        self.batch_size = 512
        self.alpha = 0.6
        self.beta = 0.4
        self.use_prioritized_experience_replay = use_prioritized_experience_replay
        if self.use_prioritized_experience_replay:
            self.prioritized_replay_buffer = PrioritizedReplayBuffer(self.buffer_size, alpha=self.alpha,
                                                                     max_priority=20.)
        else:
            self.replay_buffer = ReplayBuffer(self.buffer_size, save_return=True)

        self.margin_constant = 2

        with tf.variable_scope(self.scope_name):

            self._build_net()

            self.build_model_saver(self.scope_name)

    def _build_net(self):

        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')
        self.r = tf.placeholder(tf.float32, [None, ], name='r')
        self.a = tf.placeholder(tf.int32, [None, ], name='a')
        self.done = tf.placeholder(tf.float32, [None, ], name='done')
        self.return_value = tf.placeholder(tf.float32, [None, ], name='return')
        self.important_sampling_weight_ph = tf.placeholder(tf.float32, [None], name="important_sampling_weight")

        self.q_eval = self._build_q_net(self.s, self.n_actions, variable_scope="eval_net")
        self.q_next = self._build_q_net(self.s_, self.n_actions, variable_scope="target_net")

        t_params = scope_vars(absolute_scope_name("target_net"))
        e_params = scope_vars(absolute_scope_name("eval_net"))

        with tf.variable_scope('hard_replacement'):
            self.target_replace_op = tf.group([tf.assign(t, e) for t, e in zip(t_params, e_params)])

        with tf.variable_scope('soft_update'):
            self.update_target_q = self.__make_update_exp__(e_params, t_params)

        with tf.variable_scope('q_target'):
            self.td0_q_target = tf.stop_gradient(
                self.r + self.gamma * (1. - self.done) * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_'))

            target_action = tf.argmax(self.q_eval, axis=-1, output_type=tf.int32)
            target_a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), target_action],
                                        axis=1)
            target_q_sa = tf.gather_nd(params=self.q_next,
                                       indices=target_a_indices)
            self.double_dqn_target = tf.stop_gradient(self.r + self.gamma * (1. - self.done) * target_q_sa)

            self.montecarlo_target = self.return_value

        with tf.variable_scope('q_eval'):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)

        with tf.variable_scope('loss'):
            self._build_loss()

            self._pick_loss()

        with tf.variable_scope('train'):
            self._train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss, var_list=e_params)

    def _pick_loss(self):
        self.loss = self.double_dqn_loss
        self.priority_values = self.doubel_dqn_error

    def _build_loss(self):

        if self.use_prioritized_experience_replay:

            self.dqn_loss = tf.reduce_mean(
                self.important_sampling_weight_ph * tf.squared_difference(self.td0_q_target, self.q_eval_wrt_a,
                                                                          name='TD0_loss'))

            self.double_dqn_loss = tf.reduce_mean(
                self.important_sampling_weight_ph * tf.squared_difference(self.double_dqn_target, self.q_eval_wrt_a,
                                                                          name='Double_DQN_error'))
        else:

            self.dqn_loss = tf.reduce_mean(tf.squared_difference(self.td0_q_target, self.q_eval_wrt_a, name='TD0_loss'))

            self.double_dqn_loss = tf.reduce_mean(tf.squared_difference(self.double_dqn_target, self.q_eval_wrt_a,
                                                                        name='Double_DQN_error'))

        self.montecarlo_loss = tf.reduce_mean(tf.squared_difference(self.montecarlo_target, self.q_eval_wrt_a,
                                                                    name='MonteCarlo_error'))

        self.td0_error = tf.abs(self.td0_q_target - self.q_eval_wrt_a)
        self.doubel_dqn_error = tf.abs(self.double_dqn_target - self.q_eval_wrt_a)
        self.montecarlo_error = tf.abs(self.montecarlo_target - self.q_eval_wrt_a)

        margin_diff = tf.one_hot(self.a, self.n_actions, on_value=0., off_value=1.,
                                 dtype=tf.float32) * self.margin_constant
        self.margin_loss = tf.reduce_mean(
            tf.reduce_max(self.q_eval + margin_diff, axis=1, keepdims=False) - self.q_eval_wrt_a)
        self.mse_margin_loss = tf.reduce_mean(
            tf.squared_difference(tf.reduce_max(self.q_eval + margin_diff, axis=1, keepdims=False), self.q_eval_wrt_a))

    def _build_q_net(self, state, n_actions, variable_scope):
        with tf.variable_scope(variable_scope):
            fc1 = tf.layers.dense(state, units=self.n_features, activation=tf.nn.relu, name='fc1')
            q_out = tf.layers.dense(fc1, units=n_actions, name='q')
            return q_out

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
        if self.use_prioritized_experience_replay:
            add_episode(self.prioritized_replay_buffer, new_trajectory, gamma=self.gamma)
        else:
            add_episode(self.replay_buffer, new_trajectory, gamma=self.gamma)

    def get_action(self, sess, obs, is_test=False, other_info=None):
        if is_test:
            discrete_action = self.greedy_action(sess, obs)
        else:
            discrete_action = self.choose_action(sess, obs)

        other_action_info = {
            "learning_action": discrete_action
        }
        return 3 * discrete_action, other_action_info

    def choose_action(self, sess, observation):

        observation = observation[np.newaxis, :]
        if np.random.uniform() < self.epsilon:
            action = np.random.randint(0, self.n_actions)
        else:

            actions_value = sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value, axis=1)[0]

        return action

    def greedy_action(self, sess, single_observation):
        observation = single_observation[np.newaxis, :]
        actions_value = sess.run(self.q_eval, feed_dict={self.s: observation})
        greedy_action = np.argmax(actions_value, axis=1)[0]
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
                sess.run(self.update_target_q)
        else:

            if self.epoch % self.replace_target_iter == 0:
                sess.run(self.target_replace_op)

    def train(self, sess):
        self.update_target(sess)

        self.epoch += 1
        if not self._is_exploration_enough(self.batch_size):
            return False, [0, 0, 0, 0], 0, 0

        if self.use_prioritized_experience_replay:

            loss, montecarlo_loss, q_eval, returns = self.train_prioritized(sess)
        else:

            loss, montecarlo_loss, q_eval, returns = self.train_normal(sess)

        if self.epoch % self.epsilon_dec_iter == 0:
            self.epsilon = max(self.epsilon - self.epsilon_dec, self.epsilon_min)
            print("update epsilon:", self.epsilon)
        return True, [loss, montecarlo_loss, q_eval, returns], self.get_memory_returns(), self.epsilon

    def train_prioritized(self, sess):
        loss, q_eval, returns, montecarlo_loss = 0, 0, 0, 0
        for idx in range(1):
            sample_indices = self.prioritized_replay_buffer.make_index(self.batch_size)
            obs, act, rew, obs_next, done, dis_2_end, returns, weights, ranges = self.prioritized_replay_buffer.sample_index(
                sample_indices)
            _, loss, q_eval, montecarlo_loss, priority_values = sess.run(
                [self._train_op, self.loss, self.q_eval_wrt_a, self.montecarlo_loss, self.priority_values],
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
        loss, q_eval, returns, montecarlo_loss = 0, 0, 0, 0
        for idx in range(1):
            sample_index = self.replay_buffer.make_index(self.batch_size)
            obs, act, rew, obs_next, done, dis_2_end, returns = self.replay_buffer.sample_index(
                sample_index)
            _, loss, q_eval, montecarlo_loss = sess.run(
                [self._train_op, self.loss, self.q_eval_wrt_a, self.montecarlo_loss],
                feed_dict={
                    self.s: obs,
                    self.a: act,
                    self.r: rew,
                    self.s_: obs_next,
                    self.done: done,
                    self.return_value: returns,
                })
        return loss, montecarlo_loss, np.average(q_eval), np.average(returns)
