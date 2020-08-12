import os
import sys
sys.path.append("../")
import numpy as np
import tensorflow as tf
from agents.common.common import scope_vars, absolute_scope_name
from simulation_env.multiuser_env_cmdp import run_env, CMDPAgent, MultiUserCMDPEnv
from tensorflow.contrib.layers.python.layers import initializers

from replay_buffer.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from replay_buffer.utils import add_episode
import argparse
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class ConstrainedPPO(CMDPAgent):

    def init_parameters(self, sess):
        if self.has_target_net:
            super(CMDPAgent, self).init_parameters(sess)

            sess.run(self.a_target_replace_op)

    def __init__(self, user_num, n_actions, cvr_n_features, ppo_n_features, init_roi, budget, use_budget_control,
                 use_prioritized_experience_replay,
                 max_trajectory_length,
                 update_times_per_train=1, use_predict_cvr=False):
        self.user_num = user_num
        self.use_budget_control = use_budget_control
        self.update_times_per_train = update_times_per_train
        self.n_actions = n_actions
        self.action_dim = 1
        self.cvr_n_features = cvr_n_features
        self.ppo_n_features = ppo_n_features
        self.lr = 0.001
        self.use_predict_cvr = use_predict_cvr

        self.user_based_adjust_times = 40
        self.epsilon = 0.4
        self.epsilon_min = 0.05

        self.epsilon_dec = 0.1
        self.epsilon_dec_iter = 5000 // self.user_based_adjust_times
        self.epsilon_dec_iter_min = 500 // self.user_based_adjust_times

        self.epsilon_clip = 0.2
        self.lam = 0.5
        self.update_step = 1
        self.kl_target = 0.01
        self.gamma = 1.
        self.method = 'clip'

        self.policy_logvar = 1e-7

        self.replace_target_iter = 1
        self.soft_update_iter = 1
        self.softupdate = False

        self.scope_name = "CPPO-model"

        self.epoch = 0

        self.cvr_buffer_size = 1000 * max_trajectory_length
        self.cvr_batch_size = 512
        self.cvr_replay_buffer = ReplayBuffer(self.cvr_buffer_size, save_return=False)

        self.alpha = 0.6
        self.beta = 0.4
        self.use_prioritized_experience_replay = use_prioritized_experience_replay

        self.ppo_buffer_size = 1000 * max_trajectory_length

        self.ppo_batch_size = 250
        if self.use_prioritized_experience_replay:
            self.prioritized_replay_buffer = PrioritizedReplayBuffer(self.ppo_buffer_size, alpha=self.alpha,
                                                                     max_priority=20.)
        else:
            self.replay_buffer = ReplayBuffer(self.ppo_buffer_size, save_return=True)

        with tf.variable_scope(self.scope_name):

            self._build_net()

            self.build_model_saver(self.scope_name)

    def _build_cvr_net(self, state, variable_scope, reuse=False):
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
            cvr_out = tf.sigmoid(tf.layers.dense(fc3, units=1, name='cvr',
                                                 kernel_initializer=initializers.xavier_initializer()))
            return cvr_out

    def _build_action_net(self, state, variable_scope):
        with tf.variable_scope(variable_scope):
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
            fc3 = tf.layers.dense(fc2, units=n_features // 4, activation=tf.nn.relu, name='fc3',
                                  kernel_initializer=initializers.xavier_initializer())
            a_prob = tf.layers.dense(fc3, self.n_actions, tf.nn.softmax,
                                     kernel_initializer=initializers.xavier_initializer())
        return a_prob

    def _build_q_net(self, state, variable_scope, reuse=False):

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
            fc3 = tf.layers.dense(fc2, units=n_features // 4, activation=tf.nn.relu, name='fc3',
                                  kernel_initializer=initializers.xavier_initializer())
            v = tf.layers.dense(fc3, 1, kernel_initializer=initializers.xavier_initializer())
        return v[:, 0]

    def __make_update_exp__(self, vals, target_vals):
        polyak = 1.0 - 1e-2
        expression = []
        for var, var_target in zip(sorted(vals, key=lambda v: v.name), sorted(target_vals, key=lambda v: v.name)):
            expression.append(var_target.assign(polyak * var_target + (1.0 - polyak) * var))
        expression = tf.group(*expression)
        return expression

    def _build_net(self):

        self.s_cvr = tf.placeholder(tf.float32, [None, self.cvr_n_features], name='s_cvr')
        self.cvr = tf.placeholder(tf.float32, [None, ], name='r')

        self.s = tf.placeholder(tf.float32, [None, self.ppo_n_features], name='s')
        self.s_ = tf.placeholder(tf.float32, [None, self.ppo_n_features], name='s_')
        self.r = tf.placeholder(tf.float32, [None, ], name='r')
        self.a = tf.placeholder(tf.int32, [None, ], name='a')
        self.adv = tf.placeholder(tf.float32, [None, ], name='advantage')
        self.gamma = 1.
        self.done = tf.placeholder(tf.float32, [None, ], name='done')
        self.return_value = tf.placeholder(tf.float32, [None, ], name='return')
        self.important_sampling_weight_ph = tf.placeholder(tf.float32, [None], name="important_sampling_weight")

        self.cvr_net = self._build_cvr_net(self.s_cvr, variable_scope="cvr_net")
        self.predicted_cvr = self.cvr_net[:, 0]
        self.a_eval = self._build_action_net(self.s, variable_scope="actor_eval_net")
        self.a_target = self._build_action_net(self.s, variable_scope="actor_target_net")
        self.critic = self._build_q_net(self.s, variable_scope="eval_q_net")

        ae_params = scope_vars(absolute_scope_name("actor_eval_net"))
        at_params = scope_vars(absolute_scope_name("actor_target_net"))

        e_gmv_params = scope_vars(absolute_scope_name("eval_q_net"))
        cvr_params = scope_vars(absolute_scope_name("cvr_net"))

        with tf.variable_scope('hard_replacement'):
            self.a_target_replace_op = tf.group([tf.assign(t, e) for t, e in zip(at_params, ae_params)])

        with tf.variable_scope('loss'):
            self.cvr_loss = tf.reduce_mean(tf.squared_difference(self.predicted_cvr, self.cvr))

            self._build_loss()

            self._pick_loss()

        with tf.variable_scope('train'):
            self._train_cvr_op = tf.train.AdamOptimizer(self.lr).minimize(self.cvr_loss, var_list=cvr_params)
            self._train_ppo_critic_op = tf.train.AdamOptimizer(self.lr).minimize(self.critic_loss)
            self._train_ppo_actor_op = tf.train.AdamOptimizer(self.lr).minimize(self.actor_loss)

    def _pick_loss(self):
        self.has_target_net = True
        self.critic_loss = self.closs

        self.actor_loss = self.aloss

    def _build_loss(self):
        with tf.variable_scope('critic'):
            self.c_loss = self.return_value - self.critic
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
        cvr_trajectory = other_info["cvr"]
        for ele in cvr_trajectory:
            state, cvr = ele
            self.cvr_replay_buffer.add(state, 0, cvr, state, 0, 0, 0)

    def experience_cmdp(self, new_trajectory, other_info=None):
        if self.use_prioritized_experience_replay:
            add_episode(self.prioritized_replay_buffer, new_trajectory, gamma=self.gamma)
        else:
            add_episode(self.replay_buffer, new_trajectory, gamma=self.gamma)

    def get_agent_name(self):
        return self.scope_name

    def get_action(self, sess, obs, is_test=False, other_info=None):
        item_price = other_info["proxy_ad_price"]
        ground_truth_cvr = other_info["cvr"]
        user_alpha = other_info["user_alpha"]
        roi_thr = other_info["roi_thr"]

        observations = obs[np.newaxis, :]
        cvr = sess.run(self.predicted_cvr, feed_dict={
            self.s_cvr: observations
        })[0]
        if self.use_predict_cvr:
            bid = cvr * item_price / roi_thr
        else:
            bid = ground_truth_cvr * item_price / roi_thr
        return bid, {"cvr_over_estimate": [user_alpha, ground_truth_cvr, cvr]}

    def get_cmdp_action(self, sess, obs, is_test=False, other_info=None):
        if is_test:
            discrete_action = self.__greedy__(sess, obs)
        else:
            discrete_action = self.__epsilon_greedy__(sess, obs)

        return discrete_action

    def __greedy__(self, sess, observation):
        s = observation[np.newaxis, :]

        prob_weights = sess.run(self.a_eval, feed_dict={self.s: s})
        greedy_action = np.argmax(prob_weights, axis=1)[0]

        return greedy_action

    def __epsilon_greedy__(self, sess, observation):
        if np.random.uniform() < self.epsilon:

            action = np.random.randint(0, self.n_actions)
        else:
            action = self.__greedy__(sess, observation)
        return action

    def _is_exploration_enough(self, buffer, min_pool_size):
        return len(buffer) >= min_pool_size

    def train_cvr(self, sess):
        if not self._is_exploration_enough(self.cvr_replay_buffer, self.cvr_batch_size):
            return False, [0, 0, 0]

        cvr_loss, predicted_cvrs, cvr_targets = 0, 0, 0
        for idx in range(self.update_times_per_train):
            sample_indices = self.cvr_replay_buffer.make_index(self.cvr_batch_size)
            obs, act, cvr_targets, obs_next, done, dis_2_end, returns = self.cvr_replay_buffer.sample_index(
                sample_indices)

            _, cvr_loss, predicted_cvrs = sess.run(
                [self._train_cvr_op, self.cvr_loss, self.predicted_cvr],
                feed_dict={
                    self.s_cvr: obs,
                    self.cvr: cvr_targets
                }
            )
        return True, [cvr_loss, np.average(predicted_cvrs), np.average(cvr_targets)]

    def get_memory_returns(self):
        if self.use_prioritized_experience_replay:
            return self.prioritized_replay_buffer.current_mean_return
        else:
            return self.replay_buffer.current_mean_return

    def update_target(self, sess):
        if self.epoch % self.replace_target_iter == 0:
            sess.run(self.a_target_replace_op)

    def train(self, sess):
        if self.has_target_net:
            self.update_target(sess)
        self.epoch += 1

        buffer = self.prioritized_replay_buffer if self.use_prioritized_experience_replay else self.replay_buffer
        if not self._is_exploration_enough(buffer, self.ppo_batch_size):
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
        loss, montecarlo_loss, q_eval, returns = 0, 0, 0, 0
        for idx in range(self.update_times_per_train):
            sample_indices = self.prioritized_replay_buffer.make_index(self.ppo_batch_size)
            obs, act, rew, obs_next, done, dis_2_end, returns, weights, ranges = self.prioritized_replay_buffer.sample_index(
                sample_indices)
            _, loss, montecarlo_loss, q_eval, \
            priority_values = sess.run(
                [self._train_ppo_op, self.loss, self.montecarlo_loss, self.q_eval_wrt_a,
                 self.priority_values],
                feed_dict={
                    self.s: obs,
                    self.a: act,
                    self.r: rew,
                    self.s_: obs_next,
                    self.done: done,
                    self.return_value: returns,
                    self.important_sampling_weight_ph: weights,
                })

            priorities = priority_values + 1e-6
            self.prioritized_replay_buffer.update_priorities(sample_indices, priorities)
        return loss, montecarlo_loss, np.average(q_eval), np.average(returns)

    def train_normal(self, sess):
        loss, montecarlo_loss, q_eval, returns = 0, 0, 0, 0
        for idx in range(self.update_times_per_train):

            sample_indices = self.replay_buffer.make_index(self.ppo_batch_size)

            obs, act, rew, obs_next, done, dis_2_end, returns = self.replay_buffer.sample_index(
                sample_indices)

            adv = sess.run(self.advantage, {self.s: obs, self.return_value: returns})

            _, montecarlo_loss, q_eval = sess.run(
                [self._train_ppo_critic_op, self.critic_loss, self.critic],
                feed_dict={
                    self.s: obs,
                    self.a: act,
                    self.adv: adv,
                    self.r: rew,
                    self.s_: obs_next,
                    self.done: done,
                    self.return_value: returns,
                })
            if self.method == 'kl_pen':
                for _ in range(self.update_step):
                    _, kl, loss = sess.run(
                        [self._train_ppo_actor_op, self.kl_mean, self.actor_loss],
                        feed_dict={
                            self.adv: adv,
                            self.s: obs,
                            self.a: act,
                            self.r: rew,
                            self.done: done,
                        })
                    if kl > 4 * self.kl_target:
                        break
                if kl < self.kl_target / 1.5:
                    self.lam /= 2
                elif kl > self.kl_target * 1.5:
                    self.lam *= 2
                self.lam = np.clip(self.lam, 1e-4, 10)
            else:

                for _ in range(self.update_step):
                    _, loss = sess.run(
                        [self._train_ppo_actor_op, self.actor_loss],
                        feed_dict={
                            self.adv: adv,
                            self.s: obs,
                            self.a: act,
                            self.r: rew,
                            self.done: done,
                            self.return_value: returns,

                        })

        return loss, montecarlo_loss, np.average(q_eval), np.average(returns)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A description of what the program does.")
    parser.add_argument('--seed', default=1, type=int, help='the seed used in the program.')
    parser.add_argument('--user_num', default=1000, type=int, help='the user number of the simulation env.')
    parser.add_argument('--budget', default=12000., type=float, help='the advertising budget of an advertiser.')
    parser.add_argument('--init_cpr_thr', default=2.7647406575960796, type=float, help='the init roi_thr.')
    args = parser.parse_args()
    seed = args.seed
    user_num = args.user_num
    # init_cpr_thr = 3.3849124829058534
    init_cpr_thr = args.init_cpr_thr
    budget = args.budget

    MultiUserCMDPEnv.seed(seed)
    train = True
    use_predict_cvr = False
    user_max_request_time = 7
    cvr_n_features = 61
    ppo_n_features = 37
    update_times_per_train = 10
    action_dim = 1
    use_budget_control = True
    use_prioritized_replay = False
    agent = ConstrainedPPO(
        user_num=user_num,
        n_actions=len(MultiUserCMDPEnv.cmdp_lambda),
        cvr_n_features=cvr_n_features,
        ppo_n_features=ppo_n_features,
        init_roi=init_cpr_thr,
        budget=budget,
        use_budget_control=use_budget_control,
        use_prioritized_experience_replay=use_prioritized_replay,
        max_trajectory_length=user_max_request_time,
        update_times_per_train=update_times_per_train,
        use_predict_cvr=use_predict_cvr
    )

    if train:
        run_env(agent=agent,
                user_num=user_num, training_episode=1000, training_log_interval=1, test_interval_list=[100, 5],
                test_round=1, seed=seed, init_roi_th=init_cpr_thr,
                use_prioritized_replay=use_prioritized_replay,
                budget=budget,
                use_budget_control=use_budget_control,
                user_max_request_time=user_max_request_time
                )
