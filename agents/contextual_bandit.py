import os
import sys
sys.path.append("../")
import tensorflow as tf
from simulation_env.multiuser_env import run_env, eval_env, MultiUserEnv, PIDAgent
from simulation_env.multiuser_env import CvrAgent
from replay_buffer.replay_buffer import ReplayBuffer
from tensorflow.contrib.layers.python.layers import initializers
from agents.common.common import scope_vars, absolute_scope_name
import numpy as np
import argparse
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class ContextualBandit(PIDAgent, CvrAgent):

    def __init__(self, user_num, n_features, init_roi, budget, use_budget_control, max_trajectory_length,
                 update_times_per_train=1, ):
        PIDAgent.__init__(self, init_roi=init_roi, default_alpha=1, budget=budget, integration=1)
        self.user_num = user_num
        self.use_budget_control = use_budget_control
        self.update_times_per_train = update_times_per_train
        self.n_actions = 1
        self.n_features = n_features
        self.lr = 0.001

        self.scope_name = "MyopicGreedy-model"

        self.epoch = 0

        self.buffer_size = 1000 * max_trajectory_length
        self.batch_size = 512
        self.replay_buffer = ReplayBuffer(self.buffer_size, save_return=False)

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
            fc1 = tf.layers.dense(state, units=n_features, activation=tf.nn.relu, name='fc1')

            cvr_out = tf.sigmoid(tf.layers.dense(fc1, units=1, name='cvr'))
            return cvr_out

    def _build_net(self):

        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')
        self.cvr = tf.placeholder(tf.float32, [None, ], name='r')

        self.cvr_net = self._build_cvr_net(self.s, variable_scope="cvr_net")
        self.predicted_cvr = self.cvr_net[:, 0]

        cvr_params = scope_vars(absolute_scope_name("cvr_net"))

        with tf.variable_scope('loss'):
            self.cvr_loss = tf.reduce_mean(tf.squared_difference(self.predicted_cvr, self.cvr))

        with tf.variable_scope('train'):
            self._train_op = tf.train.AdamOptimizer(self.lr).minimize(self.cvr_loss, var_list=cvr_params)

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
            self.replay_buffer.add(state, 0, cvr, state, 0, 0, 0)

    def get_action(self, sess, obs, is_test=False, other_info=None):
        item_price = other_info["proxy_ad_price"]
        ground_truth_cvr = other_info["cvr"]
        user_alpha = other_info["user_alpha"]
        if self.use_budget_control:
            roi_thr = self.get_roi_threshold()
        else:
            roi_thr = self.init_roi

        observations = obs[np.newaxis, :]
        cvr = sess.run(self.predicted_cvr, feed_dict={
            self.s: observations
        })[0]

        bid = ground_truth_cvr * item_price / roi_thr
        return bid, {"cvr_over_estimate": [user_alpha, ground_truth_cvr, cvr]}

    def _is_exploration_enough(self, min_pool_size):
        return len(self.replay_buffer) >= min_pool_size

    def train(self, sess):
        self.epoch += 1

        if not self._is_exploration_enough(self.batch_size):
            return False, [0, 0, 0]

        cvr_loss, predicted_cvrs, cvr_targets = 0, 0, 0
        for idx in range(self.update_times_per_train):
            sample_indices = self.replay_buffer.make_index(self.batch_size)
            obs, act, cvr_targets, obs_next, done, dis_2_end, returns = self.replay_buffer.sample_index(
                sample_indices)

            _, cvr_loss, predicted_cvrs = sess.run(
                [self._train_op, self.cvr_loss, self.predicted_cvr],
                feed_dict={
                    self.s: obs,
                    self.cvr: cvr_targets
                }
            )
        return True, [cvr_loss, np.average(predicted_cvrs), np.average(cvr_targets)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A description of what the program does.")
    parser.add_argument('--seed', default=1, type=int, help='the seed used in the program.')
    parser.add_argument('--user_num', default=1000, type=int, help='the user number of the simulation env.')
    parser.add_argument('--budget', default=12000., type=float, help='the advertising budget of an advertiser.')
    parser.add_argument('--init_cpr_thr', default=6., type=float, help='the init roi_thr.')
    args = parser.parse_args()
    seed = args.seed
    user_num = args.user_num
    # init_cpr_thr = 3.3849124829058534
    init_cpr_thr = args.init_cpr_thr
    budget = args.budget

    MultiUserEnv.seed(seed)
    train = True
    user_max_request_time = 7
    n_features = 61
    update_times_per_train = 1
    use_budget_control = True
    use_prioritized_replay = False
    agent = ContextualBandit(
        user_num=user_num,
        n_features=n_features,
        init_roi=init_cpr_thr,
        budget=budget,
        use_budget_control=use_budget_control,
        max_trajectory_length=user_max_request_time,
        update_times_per_train=update_times_per_train
    )
    if train:
        run_env(agent=agent,
                user_num=user_num, training_episode=1000, training_log_interval=1, test_interval_list=[10, 5],
                test_round=1, seed=seed, init_roi_th=init_cpr_thr,
                use_prioritized_replay=use_prioritized_replay,
                budget=budget,
                use_budget_control=use_budget_control,
                user_max_request_time=user_max_request_time
                )
