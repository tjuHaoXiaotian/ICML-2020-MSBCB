import copy
import os
import random
import time
from decimal import Decimal

import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from gym import spaces

from simulation_env.utils import dump, reload_data
from plot_util.tf_log import LogScalar, LogHistogram


def cosine_similarity(item_1, item_2):
    item_1 = np.asarray(item_1)
    item_2 = np.asarray(item_2)
    return (np.sum(item_1 * item_2, dtype=np.float32)) / (
            np.sqrt(np.sum(item_1 * item_1, dtype=np.float32)) * np.sqrt(np.sum(item_2 * item_2, dtype=np.float32)))


def normalization(vector):
    return vector / np.sqrt(np.sum(vector * vector, dtype=np.float32))


def find_closest_alpha(value, candidate_alphas):
    idx = np.argmin(np.abs(candidate_alphas - value))
    return idx


class AgentInterface(object):
    def get_action(self, sess, obs, is_test=False, other_info=None):
        pass

    def experience(self, new_trajectory, other_info=None):
        pass

    def train(self, sess):
        pass

    def save(self, sess, path, step):
        pass

    def restore(self, sess, path):
        pass


class CvrAgent(AgentInterface):
    pass


class RandomAgent(AgentInterface):
    def __init__(self):
        self.action_space = spaces.Box(low=np.array(MultiUserEnv.bid_min), high=np.array(MultiUserEnv.bid_max),
                                       shape=None)
        self.n_actions = 2

    def get_action(self, sess, obs, is_test=False, other_info=None):
        return self.action_space.sample(), {}


class LearningAgent(AgentInterface):

    def train(self, sess):
        return False, [0, 0, 0, 0], 0, 0

    def init_parameters(self, sess):
        print("Copy parameters of current net to target net...")


class LearningBidAgent(LearningAgent):
    def train(self, sess):
        return False, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 0, 0


class PIDAgent(object):
    def __init__(self, init_roi, default_alpha, budget, integration=2):
        self.init_roi = init_roi
        self._pid_alpha_ = default_alpha
        self.budget = budget
        self.integration = integration

        self.k1 = 0.1
        self.k2 = 0.1
        self.episode_costs = []
        self.current_episode_explored_user = {}
        self.alpha_min = 0.01
        self.alpha_max = 10

    def reset_epsilon(self, reset_val, epsilon_dec_iter):
        self.epsilon = reset_val
        self.epsilon_dec_iter = epsilon_dec_iter

    def record_cost(self, episode_nm, episode_cost):
        self.episode_costs.append(episode_cost)

        p_ratio = episode_cost / self.budget
        i_ratio = np.sum(self.episode_costs[-self.integration:], dtype=np.float32) / (
                self.budget * len(self.episode_costs[-self.integration:]))
        before = self.get_roi_threshold()

        self._pid_alpha_ = self._pid_alpha_ * (1 + self.k1 * (p_ratio - 1) + self.k2 * (i_ratio - 1))

        print("{}: update roi_thr episodically ({}->{}, cost={}/{})...".format(episode_nm, before,
                                                                               self.get_roi_threshold(),
                                                                               episode_cost, self.budget))

    def get_roi_threshold(self):
        return np.maximum(self.init_roi * self._pid_alpha_, 1e-6)

    def explore_user(self, user_idx):
        self.current_episode_explored_user[user_idx] = True

    def is_user_selected(self, user_idx):
        return self.current_episode_explored_user.get(user_idx, False)

    def reset(self):
        self.current_episode_explored_user = {}


class MultiUserEnv(gym.Env):
    bid_min = 0.
    bid_max = 30.

    @staticmethod
    def seed(seed):
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        tf.set_random_seed(seed)
        print("input seed:", seed)
        graph = tf.get_default_graph()
        print("graph.seed:", graph.seed)

    @staticmethod
    def seed_env(seed):
        np.random.seed(seed)

    def __init__(self, budget, init_roi_th=1., render=False, user_num=2, topic_num=10, item_num=2000,
                 user_dynamic_decay=1.,
                 item_quality_ratio=0, seed=12345, user_max_request_time=7, ):
        self.seed_env(1)
        self.render_frame = render
        self.user_num = user_num
        self.topic_num = topic_num
        self.item_num = item_num
        self.budget = budget
        self.user_max_request_time = user_max_request_time
        self.channel_num = 4
        assert self.channel_num == 4
        self.item_quality_ratio = item_quality_ratio
        self.user_dynamic_decay = user_dynamic_decay
        self.roi_th = init_roi_th

        self.action_space = spaces.Box(low=0., high=30., shape=[1])

        self.init_env()
        self.env_start_time = time.time()
        self.seed_env(seed)

    def init_env(self):
        self.proxy_ad_idx = 0

        self.__init_user__()

        self.__init_items__()

        self.epoch = 0

        self.init_other()

    def init_other(self):
        pass

    def __init_user__(self):
        self.user_interest_vectors = np.random.uniform(0.0, 1, size=[self.user_num, self.topic_num], )

        self.channel_visiting_frequency = np.array([
            [0.5, 0.5, 0, 0], [0.5, 0, 0.5, 0], [0.5, 0, 0, 0.5], [0, 0.5, 0, 0.5],
            [0.5, 0.3, 0, 0.2], [0.5, 0, 0.2, 0.3]
        ])
        __generate_channel_frequency_idx__ = np.random.randint(low=0, high=6, size=[self.user_num])

        self.user_channel_visiting_frequencies = self.channel_visiting_frequency[__generate_channel_frequency_idx__, :]

        self.user_request_matrix = np.zeros(shape=[self.user_num, self.user_max_request_time], dtype=np.int32)
        for uidx in range(self.user_num):
            channel = np.argmax(np.random.multinomial(n=1, pvals=self.user_channel_visiting_frequencies[uidx],
                                                      size=self.user_max_request_time), axis=-1)

            self.user_request_matrix[uidx] = channel

        self.user_dynamic_norm = 8

        self.user_dynamic_p_values = [0.2, 0.2, 0.2, 0.4]
        self.user_candidate_alphas = np.asarray([-2, 0, 2, 6])
        self.user_candidate_alpha_index = {val: idx for idx, val in enumerate(self.user_candidate_alphas.tolist())}
        self.user_dynamic_alphas = self.user_candidate_alphas[
            np.argmax(np.random.multinomial(n=1, pvals=self.user_dynamic_p_values, size=self.user_num), axis=-1)]
        self.user_dynamic_types = len(self.user_candidate_alphas)

        # print("user_dynamic_alphas:", self.user_dynamic_alphas)

        self.p_user_level = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]

        self.p_user_level_map = {value: idx for idx, value in enumerate(self.p_user_level)}

        self.all_user_avg_bidprice_levels = np.random.choice(self.p_user_level, size=[self.user_num], replace=True)

    def __generate_independent_topic_set(self, num):
        base = list(range(self.topic_num))
        if len(base) > num:
            return np.asarray(base[:num], dtype=np.int32)
        else:
            append = list(range(1, self.topic_num))
            while len(base) < num:
                base += append
            return np.asarray(base[:num], dtype=np.int32)

    def __init_items__(self):
        channel_item_num = np.array([1, 1, 1, 1], dtype=np.int32) * self.item_num
        self.channel_bidprice_mean = np.array([5, 10, 15, 20], dtype=np.float32)
        self.channel_bidprice_variance = 1.

        item_topic_quality_distribution_mean = np.asarray([1, 0.8, 0.6, 0.5, 0.4, 0.5, 0.6, 0.3, 0.2, 0.9],
                                                          dtype=np.float32)
        item_topic_quality_distribution_var = np.asarray([0.1, 0.2, 0.3, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.1],
                                                         dtype=np.float32)
        item_topic_item_price_mean = np.asarray([128, 60, 150, 260, 30, 20, 40, 65, 45, 300],
                                                dtype=np.float32)
        self.proxy_ad_price = item_topic_item_price_mean[0]
        self.channel_items = []
        for channel_idx in range(self.channel_num):
            item_num = channel_item_num[channel_idx]
            item_topics = self.__generate_independent_topic_set(item_num)

            item_qualities = np.zeros(shape=[item_num], dtype=np.float32)
            item_prices = np.zeros(shape=[item_num], dtype=np.float32)
            for topic_idx in range(self.topic_num):
                ids = item_topics == topic_idx
                qualities = np.clip(np.random.normal(loc=item_topic_quality_distribution_mean[topic_idx],
                                                     scale=item_topic_quality_distribution_var[topic_idx],
                                                     size=np.sum(ids)), a_min=0, a_max=1)
                prices = np.round(np.clip(
                    np.random.normal(loc=item_topic_item_price_mean[topic_idx], scale=10, size=np.sum(ids)), a_min=0,
                    a_max=1000), decimals=2)
                item_qualities[ids] = qualities
                item_prices[ids] = prices

            item_bidprices = np.clip(
                np.random.normal(loc=self.channel_bidprice_mean[channel_idx], scale=self.channel_bidprice_variance,
                                 size=item_num), a_min=self.bid_min, a_max=self.bid_max)

            item_topics[self.proxy_ad_idx] = 0
            item_qualities[self.proxy_ad_idx] = 1
            item_bidprices[self.proxy_ad_idx] = self.channel_bidprice_mean[channel_idx] * 1.2
            item_prices[self.proxy_ad_idx] = item_topic_item_price_mean[item_topics[self.proxy_ad_idx]]

            channel_item = np.stack([item_topics, item_qualities, item_bidprices, item_prices], axis=1)
            self.channel_items.append(channel_item)

    def __reset_user__(self):
        self.user_interest_vector = self.user_interest_vectors[self.user_idx]
        self.user_dynamic_alpha = self.user_dynamic_alphas[self.user_idx]
        if self.render_frame:
            print("user_dynamic_alpha:", self.user_dynamic_alpha / self.user_dynamic_norm, self.user_interest_vector)
        self.user_cumulative_interaction_times = 0
        self.user_cumulative_cost, self.user_cumulative_gmv = 0, 0
        self.previous_action = -1
        self.previous_actions = []
        self.previous_reward = 0

        self.user_request_channels = self.user_request_matrix[self.user_idx]
        self.user_avg_bidprice_level = self.all_user_avg_bidprice_levels[self.user_idx]

    def __reset_request__(self):
        current_channel = self.user_request_channels[self.request_idx]
        candidate_items = self.channel_items[current_channel]
        self.item_topics = np.eye(self.topic_num, dtype=np.float32)[candidate_items[:, 0].astype(np.int32)]
        item_qualities = candidate_items[:, 1]
        self.item_bidprices = candidate_items[:, 2]
        self.item_prices = candidate_items[:, 3]
        I = np.dot(self.item_topics, self.user_interest_vector) / (
                np.sqrt(np.sum(self.item_topics * self.item_topics, axis=1, dtype=np.float32)) * np.sqrt(
            np.sum(self.user_interest_vector * self.user_interest_vector, dtype=np.float32)) + 1e-10)
        self.cvrs = np.maximum((1 - self.item_quality_ratio) * I + self.item_quality_ratio * item_qualities,
                               0)
        self.ctrs = np.ones_like(self.cvrs, dtype=np.float32)

        user_channel_visiting_frequency = self.user_channel_visiting_frequencies[self.user_idx]
        onehot_request_idx = [0] * self.user_max_request_time
        onehot_request_idx[self.request_idx] = 1
        onehot_cumulative_interaction_times = [0] * self.user_max_request_time
        onehot_cumulative_interaction_times[self.user_cumulative_interaction_times] = 1
        onehot_channel_idx = [0] * self.channel_num
        onehot_channel_idx[current_channel] = 1

        ctr = self.ctrs[self.proxy_ad_idx]
        cvr = self.cvrs[self.proxy_ad_idx]
        onehot_user_dynamic_alpha = [0] * self.user_dynamic_types

        onehot_user_dynamic_alpha[self.user_candidate_alpha_index[self.user_dynamic_alphas[self.user_idx]]] = 1

        previous_onehot_action = [0] * 2
        if self.previous_action != -1:
            previous_onehot_action[self.previous_action] = 1
        onehot_previous_actions = copy.deepcopy(self.previous_actions)
        onehot_previous_actions += [0] * (self.user_max_request_time - len(self.previous_actions))

        onehot_user_bidprice_level = [0] * len(self.p_user_level)
        onehot_user_bidprice_level[self.p_user_level_map[self.user_avg_bidprice_level]] = 1
        probability_of_not_buying = self.__cal_cost_prob__(epoch_cvrs=self.epoch_cvrs)
        self.state = np.asarray(

            [self.user_idx] +
            self.user_interest_vector.tolist() +
            onehot_user_bidprice_level +
            onehot_request_idx +
            onehot_cumulative_interaction_times +
            user_channel_visiting_frequency.tolist() +
            onehot_user_dynamic_alpha +
            onehot_previous_actions +
            previous_onehot_action + [self.previous_reward] +
            onehot_channel_idx + [
                np.clip(self.user_avg_bidprice_level * self.channel_bidprice_mean[current_channel], a_min=self.bid_min,
                        a_max=self.bid_max)] +
            [
                probability_of_not_buying,
                probability_of_not_buying * cvr,

                self.user_cumulative_gmv / self.proxy_ad_price,
                self.user_cumulative_cost,
            ],
            dtype=np.float32)

    def reset_others(self):
        pass

    def reset(self):
        self.user_idx = 0
        self.request_idx = 0
        self.epoch_cvrs = []

        self.__reset_user__()
        self.__reset_request__()
        USER_END, self.END = False, False
        self.info = {
            "epoch": self.epoch,
            "user_idx": self.user_idx,
            "request_idx": self.request_idx,
            "user_alpha": self.user_dynamic_alpha,
            "ctr": self.ctrs[self.proxy_ad_idx],
            "cvr": self.cvrs[self.proxy_ad_idx],
            "bid": 0,
            "second_price": 0,
            "action": 0,

            "end": self.END,
            "user_end": USER_END,
            "proxy_ad_price": self.proxy_ad_price,
            "probability_of_not_buying": self.__cal_cost_prob__(epoch_cvrs=self.epoch_cvrs)
        }
        if self.render_frame:
            self.render()

        self.reset_others()
        return self.state, [USER_END, self.END], self.info

    def __calculate_winner__(self, ecpms, topK=1):
        sorted_indexes = np.argsort(-ecpms, axis=0)
        winner_indexes = sorted_indexes[:topK + 1]
        winner_ecpm_scores = ecpms[winner_indexes]
        winner_second_ecpm_scores = winner_ecpm_scores[1:]
        winner_pctrs = self.ctrs[winner_indexes][:-1] + 1e-6
        winner_cost_if_clicked = winner_second_ecpm_scores / winner_pctrs + 0.01
        displayed_ad_indexes = winner_indexes[:topK]
        costs_per_click = winner_cost_if_clicked
        displayed_ad_prices = self.item_prices[displayed_ad_indexes]
        return displayed_ad_indexes, costs_per_click, displayed_ad_prices, winner_second_ecpm_scores, winner_ecpm_scores[
                                                                                                      :-1]

    def __is_ad_displayed__(self, displayed_ad_indexes):

        for idx in range(len(displayed_ad_indexes)):
            if displayed_ad_indexes[idx] == self.proxy_ad_idx:
                return True, idx
        return False, -1

    def __calculate_reward__(self, displayed_ad_indexes, costs_per_click, first_ecpm_scores, second_ecpm_scores,
                             displayed_ad_prices, pctrs, pcvrs, is_test=False):

        display, expected_gmv, expected_cost, cvr, ecpm2 = 0, 0, 0, 0, 0
        USER_END = False
        if self.request_idx == self.user_max_request_time - 1:
            USER_END = True
        is_display, display_pos = self.__is_ad_displayed__(displayed_ad_indexes)

        if is_display:
            display = 1
            ecpm1 = first_ecpm_scores[display_pos]
            ecpm2 = second_ecpm_scores[display_pos]

            price = displayed_ad_prices[display_pos]
            cvr = pcvrs[display_pos]
            cost_prob = self.__cal_cost_prob__(epoch_cvrs=self.epoch_cvrs)
            expected_cost = np.asarray(Decimal(cost_prob * ecpm2).quantize(Decimal('0.00')),
                                       dtype=np.float32)

            expected_gmv = np.asarray(Decimal(cost_prob * cvr * price).quantize(Decimal('0.00')), dtype=np.float32)

            self.user_cumulative_interaction_times += 1
            self.user_cumulative_cost += expected_cost
            self.user_cumulative_gmv += expected_gmv
            self.epoch_cvrs.append(cvr)

        return expected_gmv, expected_cost, display, cvr, ecpm2, USER_END

    def __cal_cost_prob__(self, epoch_cvrs):
        size = len(epoch_cvrs)
        p = 1.
        for i in range(size):
            p *= (1 - epoch_cvrs[i])
        return p

    def step(self, action=0, is_test=False):
        if self.render_frame:
            print("user", self.user_idx, "request", self.request_idx)

        self.item_bidprices = np.clip(self.item_bidprices * self.user_avg_bidprice_level, a_min=self.bid_min,
                                      a_max=self.bid_max)

        if action is not None:
            self.item_bidprices[self.proxy_ad_idx] = action

        ecpms = self.item_bidprices * self.ctrs
        displayed_ad_indexes, costs_per_click, displayed_ad_prices, second_ecpm_scores, first_ecpm_scores = self.__calculate_winner__(
            ecpms, topK=1)

        expected_gmv, expected_cost, display, cvr, ecpm2, USER_END = self.__calculate_reward__(displayed_ad_indexes,
                                                                                               costs_per_click,
                                                                                               first_ecpm_scores,
                                                                                               second_ecpm_scores,
                                                                                               displayed_ad_prices,
                                                                                               self.ctrs[
                                                                                                   displayed_ad_indexes],
                                                                                               self.cvrs[
                                                                                                   displayed_ad_indexes],
                                                                                               is_test=is_test)

        reward = np.asarray(expected_gmv - self.roi_th * expected_cost, dtype=np.float32)

        self.previous_action = display
        self.previous_actions.append(display)
        self.previous_reward = reward

        displayed_onehot_topics = self.item_topics[displayed_ad_indexes]

        displayed_ctrs = self.ctrs[displayed_ad_indexes]

        ad_impacts = displayed_onehot_topics * np.expand_dims(displayed_ctrs,
                                                              axis=1)

        interesting_topics = np.sum(ad_impacts, axis=0, dtype=np.float32)

        self.user_interest_vector = self.user_dynamic_decay * self.user_interest_vector + self.user_dynamic_alpha / self.user_dynamic_norm * interesting_topics
        self.user_interest_vector = normalization(self.user_interest_vector)

        self.END = False
        if USER_END:
            self.user_idx += 1
            if self.render_frame:
                time.sleep(1)
                print("\n")
            self.epoch_cvrs = []
            if self.user_idx == self.user_num:
                self.user_idx -= 1
                self.END = True
                self.epoch += 1
                if self.render_frame:
                    print("========= END ========= \n")
            else:
                self.request_idx = 0
                self.__reset_user__()
                self.__reset_request__()

        else:
            self.request_idx += 1
            self.__reset_request__()
        self.info["epoch"] = self.epoch
        self.info["user_idx"] = self.user_idx
        self.info["request_idx"] = self.request_idx
        self.info["user_alpha"] = self.user_dynamic_alpha
        self.info["ctr"] = self.ctrs[self.proxy_ad_idx]
        self.info["cvr"] = self.cvrs[self.proxy_ad_idx]
        self.info["bid"] = action
        self.info["second_price"] = ecpm2
        self.info["action"] = display

        self.info["end"] = self.END
        self.info["user_end"] = USER_END
        self.info["probability_of_not_buying"] = self.__cal_cost_prob__(epoch_cvrs=self.epoch_cvrs)

        if self.render_frame:
            self.render()

        self.step_others(display, expected_gmv, expected_cost, reward)

        return self.state, [expected_gmv, expected_cost, reward], [USER_END, self.END], self.info

    def step_others(self, display, expected_gmv, expected_cost, reward):
        pass

    def render(self):
        print(self.info)

    def close(self):
        self.env_end_time = time.time()
        print("Env close, it costs {} seconds...".format(self.env_end_time - self.env_start_time))


def show_user_dynamic_ctr(path):
    all_user_ctrs = reload_data(path)
    for user_idx, ctrs in all_user_ctrs.items():
        plt.plot(ctrs)
    plt.show()
    exit()


def run_env(agent, user_num=1, training_episode=100, training_log_interval=10, test_interval_list=5, test_round=1,
            seed=12345, init_roi_th=1., use_prioritized_replay=False, budget=100.,
            use_budget_control=False, user_max_request_time=8):
    start_time = time.time()
    print("use_budget_control={}, budget={}".format(use_budget_control, budget))

    assert isinstance(agent, AgentInterface)
    BEST_TEST_RETURN = 0

    env = MultiUserEnv(init_roi_th=init_roi_th, render=False, user_num=user_num, topic_num=10, item_num=10,
                       seed=seed, user_max_request_time=user_max_request_time, budget=budget)

    __log_path = os.path.join("./exp/learning_result", agent.__class__.__name__, "train",

                              "{}_user".format(user_num),
                              "action_n={}".format(agent.n_actions),
                              "per={}".format(1 if use_prioritized_replay else 0),
                              "seed={}".format(str(seed)),
                              str(int(time.time())) + "/")
    __model_path = os.path.join(__log_path, "best_model/model")
    __avg_round = 20

    training_curves = [
        "policy_loss", "policy_entorpy",
        "q_loss", "montecarlo_loss", "q_value", "q_value_true", "q_value_diff",
        "q_gmv_loss", "montecarlo_gmv_loss", "q_gmv_value", "q_gmv_value_true", "q_gmv_value_diff",
        "q_cost_loss", "montecarlo_cost_loss", "q_cost_value", "q_cost_value_true", "q_cost_value_diff",
        "avg_buffer_return", "epsilon"
    ]
    pid_curves = ["pid_roi_threshold", "pid_cost", "pid_gmv"]
    testing_curves = ["test_avg_return", "test_avg_gmv", "test_avg_cost", "test_egmv_div_ecost", "test_avg_a-1",
                      "test_avg_gap_to_second_price"]
    learning_curves = [ele.format(__avg_round) for ele in
                       ["train_avg{}_return", "train_avg{}_gmv", "train_avg{}_cost", "train_avg{}_egmv_div_ecost",
                        "train_avg{}_a-1"]]
    cvr_curves = ["cvr_mse_loss", "avg_predicted_cvrs", "avg_cvr_targets"]

    __gpu_options = tf.GPUOptions(allow_growth=True)
    __config = tf.ConfigProto(log_device_placement=False, gpu_options=__gpu_options)

    with tf.Session(config=__config, graph=tf.get_default_graph()) as tf_sess:
        logger_scalar = LogScalar(sess=tf_sess,
                                  kws=training_curves + testing_curves + learning_curves + pid_curves + cvr_curves,
                                  log_path=__log_path, build_new_sess=False)

        testing_histogram_curves = ["test_bid_distribution"]
        learning_histogram_curves = ["train_bid_distribution"]
        logger_histogram = LogHistogram(sess=tf_sess, kws=testing_histogram_curves + learning_histogram_curves,
                                        shapes=[None, None],
                                        log_path=None, exist_writer=logger_scalar.writer)

        tf_sess.run(tf.global_variables_initializer())

        if isinstance(agent, LearningAgent):
            agent.init_parameters(sess=tf_sess)

        tf_sess.graph.finalize()

        train_returns, train_gmvs, train_costs, train_actions, train_bids = [], [], [], [], []
        test_returns, test_gmvs, test_costs, test_actions, test_bids, test_all_actions, test_gaps_to_second_price = [], [], [], [], [], [], []

        training_ep, testing_ep = 0, 0
        is_test = False

        state, [user_done, done], _info = env.reset()
        test_times = 0

        if isinstance(agent, CvrAgent):
            all_epoch_cvr_over_estimation_gmv_summary = []

        all_trajectories = []
        while training_ep < (training_episode + 1):
            if isinstance(test_interval_list, int):
                test_interval = test_interval_list
            else:
                if test_times == 0:
                    test_interval = test_interval_list[0]
                else:
                    test_interval = test_interval_list[1]

            if training_ep % test_interval == 0 and training_ep != 0:
                if testing_ep < test_round:
                    is_test = True
                else:
                    is_test = False

                    avg_gmv = np.average(test_gmvs)
                    avg_cost = np.average(test_costs)
                    avg_return = np.average(test_returns)
                    avg_action_num = np.average(test_actions)
                    if len(test_gaps_to_second_price) > 0:
                        avg_gap_to_second_price = np.average(test_gaps_to_second_price)
                        logger_scalar.log(testing_curves,
                                          [avg_return, avg_gmv, avg_cost, avg_gmv / (avg_cost + 1e-6), avg_action_num,
                                           avg_gap_to_second_price], global_step=training_ep)
                    else:
                        logger_scalar.log(testing_curves[:-1],
                                          [avg_return, avg_gmv, avg_cost, avg_gmv / (avg_cost + 1e-6), avg_action_num],
                                          global_step=training_ep)

                    bid_distribution = []
                    for bids in test_bids:
                        bid_distribution += bids

                    logger_histogram.log(testing_histogram_curves, [bid_distribution], global_step=training_ep)
                    print("ep {}: avg return={}".format(training_ep, avg_return))

                    testing_ep = 0
                    test_returns, test_gmvs, test_costs, test_actions, test_bids, test_all_actions = [], [], [], [], [], []
                    if avg_return > BEST_TEST_RETURN:
                        agent.save(sess=tf_sess, path=__model_path, step=training_ep)

                    if isinstance(agent, PIDAgent) and use_budget_control:
                        env.roi_th = agent.get_roi_threshold()

                    test_times += 1
                    if training_ep == training_episode:
                        break

            new_trajectory, new_gmv_trajectory, new_cost_trajectory, new_gmv_cost_trajectory, cvr_trajectory = [], [], [], [], []
            traj_gmv, traj_cost, traj_reward, traj_action, traj_bid, traj_gap_to_second_price = [], [], [], [], [], []
            episode_cost, episode_gmv = 0, 0
            if isinstance(agent, CvrAgent):
                cvr_over_estimation_point_summary = np.zeros(shape=[env.user_num, env.user_max_request_time],
                                                             dtype=np.int32)
                cvr_over_estimation_point_gmv_summary = np.zeros(shape=[env.user_num, env.user_max_request_time],
                                                                 dtype=np.float32)
            while not done:
                bid, other_action_info = agent.get_action(tf_sess, state, is_test=is_test, other_info=_info)
                state_tp1, [gmv, cost, reward], [user_done, done], _info = env.step(bid, is_test=is_test)
                if isinstance(agent, LearningBidAgent):
                    action = _info["action"]
                elif isinstance(agent, LearningAgent):
                    action = other_action_info["learning_action"]
                elif isinstance(agent, CvrAgent):
                    user_alpha, ground_truth_cvr, cvr = other_action_info["cvr_over_estimate"]

                    user_idx, request_idx = _info["user_idx"] - 1, _info["request_idx"]
                    if cvr > ground_truth_cvr:
                        cvr_over_estimation_point_summary[user_idx][request_idx] = 1
                    elif cvr < ground_truth_cvr:
                        pass
                    cvr_over_estimation_point_gmv_summary[user_idx][request_idx] = gmv
                    action = bid
                else:
                    action = bid

                new_trajectory.append([state, action, reward, state_tp1, done])
                new_gmv_trajectory.append([state, action, gmv, state_tp1, done])
                new_cost_trajectory.append([state, action, cost, state_tp1, done])
                new_gmv_cost_trajectory.append([state, action, [gmv, cost, reward], state_tp1, done])
                cvr_trajectory.append([state, _info["cvr"]])
                if user_done:
                    if not is_test:
                        agent.experience(new_trajectory, {
                            "gmv": new_gmv_trajectory,
                            "cost": new_cost_trajectory,
                            "gmv_and_cost": new_gmv_cost_trajectory,
                            "cvr": cvr_trajectory
                        })
                        all_trajectories.append(new_trajectory)
                    new_trajectory, new_gmv_trajectory, new_cost_trajectory, new_gmv_cost_trajectory, cvr_trajectory = [], [], [], [], []

                if use_budget_control:
                    if episode_cost < budget:
                        traj_gmv.append(gmv)
                        traj_cost.append(cost)
                        traj_reward.append(reward)
                        traj_action.append(_info["action"])
                        traj_bid.append(bid)
                        if _info["action"] == 1:
                            gap_to_second_price = bid - _info["second_price"]
                            traj_gap_to_second_price.append(gap_to_second_price)
                else:
                    traj_gmv.append(gmv)
                    traj_cost.append(cost)
                    traj_reward.append(reward)
                    traj_action.append(_info["action"])
                    traj_bid.append(bid)
                    if _info["action"] == 1:
                        gap_to_second_price = bid - _info["second_price"]
                        traj_gap_to_second_price.append(gap_to_second_price)

                episode_cost += cost
                episode_gmv += gmv

                state = state_tp1

            if is_test:
                if isinstance(agent, PIDAgent) and use_budget_control:
                    logger_scalar.log(pid_curves, [agent.get_roi_threshold(), episode_cost, episode_gmv],
                                      global_step=training_ep + testing_ep)

                    agent.record_cost(training_ep + testing_ep, episode_cost)

                testing_ep += 1

                test_returns.append(np.sum(traj_reward, dtype=np.float32))

                test_gmvs.append(np.sum(traj_gmv, dtype=np.float32))
                test_costs.append(np.sum(traj_cost, dtype=np.float32))
                test_actions.append(np.sum(traj_action, dtype=np.float32))
                test_all_actions += traj_action
                test_bids.append(traj_bid)
                if len(traj_gap_to_second_price) > 0:
                    test_gaps_to_second_price.append(np.average(traj_gap_to_second_price))
            else:
                if isinstance(agent, LearningAgent):
                    trained, [policy_loss, policy_entropy, loss, montecarlo_loss, q_eval, returns,
                              gmv_loss, gmv_montecarlo_loss, gmv_q_eval, gmv_returns,
                              cost_loss, cost_montecarlo_loss, cost_q_eval, cost_returns], \
                    avg_buffer_return, epsilon = agent.train(tf_sess)

                    if training_ep % training_log_interval == 0 and trained:
                        logger_scalar.log(training_curves, [
                            policy_loss, policy_entropy,
                            loss, montecarlo_loss, q_eval, returns, q_eval - returns,
                            gmv_loss, gmv_montecarlo_loss, gmv_q_eval, gmv_returns, gmv_q_eval - gmv_returns,
                            cost_loss, cost_montecarlo_loss, cost_q_eval, cost_returns, cost_q_eval - cost_returns,
                            avg_buffer_return, epsilon
                        ], global_step=training_ep)

                    if training_ep % 100 == 0:
                        seconds = time.time() - start_time
                        print("{}: losses=[policy={}, entropy={}, q={}] cost {} seconds ({} minutes)...".format(
                            training_ep,
                            policy_loss,
                            policy_entropy,
                            loss,
                            seconds,
                            round(seconds / 60, 2)))

                elif isinstance(agent, CvrAgent):
                    trained, [cvr_loss, avg_predicted_cvrs, avg_cvr_targets] = agent.train(tf_sess)
                    if training_ep % training_log_interval == 0 and trained:
                        logger_scalar.log(cvr_curves, [cvr_loss, avg_predicted_cvrs, avg_cvr_targets],
                                          global_step=training_ep)
                    if training_ep % 100 == 0:
                        seconds = time.time() - start_time
                        print("{}: losses=[cvr-loss={}] cost {} seconds ({} minutes)...".format(
                            training_ep,
                            cvr_loss,
                            seconds,
                            round(seconds / 60, 2)))

                training_ep += 1

                train_returns.append(np.sum(traj_reward, dtype=np.float32))
                train_gmvs.append(np.sum(traj_gmv, dtype=np.float32))
                train_costs.append(np.sum(traj_cost, dtype=np.float32))
                train_actions.append(np.sum(traj_action, dtype=np.float32))
                train_bids.append(traj_bid)
                if training_ep % training_log_interval == 0:

                    avg_gmv = np.average(train_gmvs[-__avg_round:])
                    avg_cost = np.average(train_costs[-__avg_round:])
                    avg_action_num = np.average(train_actions[-__avg_round:])
                    logger_scalar.log(learning_curves,
                                      [np.average(train_returns[-__avg_round:]), avg_gmv, avg_cost,
                                       avg_gmv / (avg_cost + 1e-6), avg_action_num],
                                      global_step=training_ep)

                    bid_distribution = []
                    for bids in train_bids[-training_log_interval:]:
                        bid_distribution += bids
                    logger_histogram.log(learning_histogram_curves, [bid_distribution], global_step=training_ep)

            state, [user_done, done], _info = env.reset()
            if training_ep % 10 == 0:
                print(training_ep)

            if isinstance(agent, CvrAgent):
                all_epoch_cvr_over_estimation_gmv_summary.append(
                    [cvr_over_estimation_point_summary, cvr_over_estimation_point_gmv_summary])
                dump(all_epoch_cvr_over_estimation_gmv_summary,
                     "{}{}".format(__log_path, "all_epoch_cvr_over_estimation_gmv_summary.pkl"))

                cvr_over_estimation_point_summary = np.zeros(shape=[env.user_num, env.user_max_request_time],
                                                             dtype=np.int32)
                cvr_over_estimation_point_gmv_summary = np.zeros(shape=[env.user_num, env.user_max_request_time],
                                                                 dtype=np.float32)
                pass

            if isinstance(agent, PIDAgent) and use_budget_control:
                agent.reset()

        dump(all_trajectories, "./training_data_{}.pkl".format(training_episode))
    env.close()


def eval_env(agent, model_path, user_num=1, seed=12345, test_epoch=2, init_roi_th=1.,
             print_log=True, use_prioritized_replay=False, budget=100.,
             use_budget_control=False, user_max_request_time=8):
    assert isinstance(agent, AgentInterface)

    env = MultiUserEnv(init_roi_th=init_roi_th, render=False, user_num=user_num, topic_num=10, item_num=10,
                       seed=seed, user_max_request_time=user_max_request_time, budget=budget)

    __avg_round = 20

    __gpu_options = tf.GPUOptions(allow_growth=True)
    __config = tf.ConfigProto(log_device_placement=False, gpu_options=__gpu_options)

    __log_path = os.path.join("./exp/learning_result", agent.__class__.__name__, "test",

                              "{}_user".format(user_num),
                              "per={}".format(1 if use_prioritized_replay else 0),
                              "seed={}".format(str(seed)),
                              str(int(time.time())) + "/")
    tf_sess = tf.Session(config=__config)
    pid_curves = ["pid_roi_threshold", "pid_cost"]
    gap_to_second_price_curves = ["gap_to_price2"]
    testing_curves = ["test_avg_return", "test_avg_gmv", "test_avg_cost", "test_egmv_div_ecost"]
    logger_scalar = LogScalar(sess=tf_sess, kws=testing_curves + pid_curves + gap_to_second_price_curves,
                              log_path=__log_path, build_new_sess=False)

    testing_histogram_curves = ["test_bid_distribution"]
    logger_histogram = LogHistogram(sess=tf_sess, kws=testing_histogram_curves,
                                    shapes=[None],
                                    log_path=None, exist_writer=logger_scalar.writer)

    agent.restore(tf_sess, path=model_path)

    tf_sess.graph.finalize()

    testing_ep = 0
    state, [user_done, done], _info = env.reset()
    test_returns, test_gmvs, test_costs, test_bids = [], [], [], []
    all_epoch_values = []

    user_gmv, user_cost, executed_actions = 0, 0, []
    user_idx = _info["user_idx"]
    # assert test_epoch == 1
    all_trajectories = []
    while testing_ep < (test_epoch):
        all_user_epoch_data = {}
        traj_gmv, traj_cost, traj_reward, traj_bid = [], [], [], []
        episode_cost = 0

        if print_log:
            traj_ctr_print, traj_cost_print, traj_ecpm2_print = [], [], []
            traj_ctr_print.append(np.round(_info["cvr"], decimals=2))
            print(testing_ep, ": action=[", sep="", end="")

        all_request_gap_to_second_prices = []
        new_trajectory = []
        while not done:
            bid, other_action_info = agent.get_action(tf_sess, state, is_test=True, other_info=_info)
            state_tp1, [gmv, cost, reward], [user_done, done], _info = env.step(bid)
            if isinstance(agent, LearningBidAgent):
                action = _info["action"]
            elif isinstance(agent, LearningAgent):
                action = other_action_info["learning_action"]
            elif isinstance(agent, CvrAgent):
                action = bid
            else:
                action = bid
            new_trajectory.append([state, action, reward, state_tp1, done])
            if user_done:
                all_trajectories.append(new_trajectory)
                new_trajectory = []

            user_gmv += gmv
            user_cost += cost
            if _info["action"] == 1:
                gap_to_second_price = bid - _info["second_price"]
                all_request_gap_to_second_prices.append(gap_to_second_price)

            if isinstance(agent, LearningBidAgent):
                action = _info["action"]
            elif isinstance(agent, LearningAgent):
                action = other_action_info["learning_action"]
            else:
                action = bid
            executed_actions.append(_info["action"])

            if print_log:
                print("{}".format(action), end=" ")
                traj_ctr_print.append(np.round(_info["cvr"], decimals=2))
                traj_cost_print.append(np.round(cost, decimals=2))

            if user_done:
                if print_log:
                    print("], buy={}".format(_info["buy"]), sep="", end=", ")
                    print("cvr={}".format(traj_ctr_print), end=", ")
                    print("cost={}".format(traj_cost_print), end="\n")

                    if not done:
                        traj_ctr_print, traj_cost_print, traj_ecpm2_print = [], [], []
                        traj_ctr_print.append(np.round(_info["cvr"], decimals=2))
                        print(testing_ep, ": action=[", sep="", end="")
                    else:
                        print("-------------- episode_cost:", episode_cost, "----------------")
                        print()
                if all_user_epoch_data.get(user_idx, None) is None:
                    user_gmv = round(user_gmv * 100)
                    user_cost = round(user_cost * 100)
                    user_roi = user_gmv / (user_cost + 1e-16)
                    all_user_epoch_data[user_idx] = [user_gmv, user_cost, user_roi, executed_actions]
                    print(user_idx, user_gmv, user_cost, executed_actions)
                user_gmv, user_cost, executed_actions = 0, 0, []
                user_idx = _info["user_idx"]

                if len(all_request_gap_to_second_prices) > 0:
                    logger_scalar.log(gap_to_second_price_curves, [np.average(all_request_gap_to_second_prices)],
                                      global_step=user_idx)
                    all_request_gap_to_second_prices = []

            if use_budget_control:
                if episode_cost < budget:
                    traj_gmv.append(gmv)
                    traj_cost.append(cost)
                    traj_reward.append(reward)
                    traj_bid.append(bid)
            else:
                traj_gmv.append(gmv)
                traj_cost.append(cost)
                traj_reward.append(reward)
                traj_bid.append(bid)

            episode_cost += cost

            state = state_tp1

        if isinstance(agent, PIDAgent) and use_budget_control:
            agent.record_cost(testing_ep, episode_cost)

            logger_scalar.log(pid_curves, [agent.get_roi_threshold(), episode_cost],
                              global_step=testing_ep)

        sum_return = np.sum(traj_reward, dtype=np.float32)
        sum_gmv = np.sum(traj_gmv, dtype=np.float32)
        sum_cost = np.sum(traj_cost, dtype=np.float32)
        test_returns.append(sum_return)
        test_gmvs.append(sum_gmv)
        test_costs.append(sum_cost)
        test_bids.append(traj_bid)

        policy_idx = testing_ep % 2 ** env.user_max_request_time
        if len(all_epoch_values) <= policy_idx:
            all_epoch_values.append([[], [], []])
        all_epoch_values[policy_idx][0].append(sum_return)
        all_epoch_values[policy_idx][1].append(sum_gmv)
        all_epoch_values[policy_idx][2].append(sum_cost)

        avg_gmv = np.average(test_gmvs[-__avg_round:])
        avg_cost = np.average(test_costs[-__avg_round:])
        logger_scalar.log(testing_curves,
                          [np.average(test_returns[-__avg_round:]), avg_gmv, avg_cost, avg_gmv / (avg_cost + 1e-6)],
                          global_step=testing_ep)

        bid_distribution = []
        for bids in test_bids[-__avg_round:]:
            bid_distribution += bids

        logger_histogram.log(testing_histogram_curves, [bid_distribution], global_step=testing_ep)

        state, [user_done, done], _info = env.reset()
        testing_ep += 1
        if testing_ep % 1000 == 0:
            print(testing_ep)

        if isinstance(agent, PIDAgent) and use_budget_control:
            agent.reset()

        dump(all_user_epoch_data, "{}{}".format(__log_path, "epoch_{}_user_values.pkl".format(testing_ep)))
        dump(all_request_gap_to_second_prices,
             "{}{}".format(__log_path, "all_request_gap_to_second_prices_{}.pkl".format(testing_ep)))

    # dump(all_trajectories, "./evaluation.pkl")
    dump(all_epoch_values, "{}{}".format(__log_path, "all_epoch_values.pkl"))
    env.close()


def enumerate_policies(size):
    if size > 1:
        tmp_result_list = enumerate_policies(size - 1)
        result_list = []
        for part in tmp_result_list:
            result_list.append([0] + part)
            result_list.append([1] + part)
        return result_list
    elif size == 1:
        result_list = [[0], [1]]
        return result_list


def test_env(seed=1, budget=12000):
    MultiUserEnv.seed(1)
    env = MultiUserEnv(init_roi_th=1., render=False, seed=seed, user_num=1000, item_num=10, user_max_request_time=7,
                       budget=budget)
    agent = lambda ob: env.action_space.sample()[0]
    all_user_ctr = {}
    for _ in range(1):
        sum_gmv, sum_cost, sum_reward = 0, 0, 0
        epoch_cost = []
        ob, [user_end, end], _info = env.reset()
        new_trajectory = []
        while not end:
            if all_user_ctr.get(_info["user_idx"], None) is None:
                all_user_ctr[_info["user_idx"]] = []
            all_user_ctr[_info["user_idx"]].append(_info["cvr"])
            a = agent(ob)

            ob, [gmv, cost, reward], [user_end, end], _info = env.step(None)
            if sum_cost + cost > budget:
                break
            epoch_cost.append(cost)
            sum_gmv += gmv
            sum_cost += cost
            sum_reward += reward

        print("GMV={}, Cost={}, Reward={}".format(sum_gmv, sum_cost, sum_reward))
    env.close()


def show_single_user_points(seed=1):
    env = MultiUserEnv(init_roi_th=1., render=False, seed=seed, user_num=1, item_num=10)
    policies = enumerate_policies(env.user_max_request_time)
    print(policies)

    all_possible_curves = {}
    all_possible_points_x, all_possible_points_y = [], []
    for possible_idx in range(len(policies)):
        current_policy = policies[possible_idx]
        epoch_cost = 0
        epoch_gmv = 0
        ob, [user_end, end], _info = env.reset()
        while not end:
            if all_possible_curves.get(possible_idx, None) is None:
                all_possible_curves[possible_idx] = []
            all_possible_curves[possible_idx].append(_info["cvr"])
            bid = 1000. if current_policy[env.request_idx] == 1 else 0
            ob, [gmv, cost, reward], [user_end, end], _info = env.step(bid)
            epoch_cost += cost
            epoch_gmv += gmv
        all_possible_points_y.append(epoch_gmv)
        all_possible_points_x.append(epoch_cost)
    env.close()

    for user_idx, cvrs in all_possible_curves.items():
        plt.plot(cvrs)
    plt.xlabel('step')
    plt.ylabel('cvr')
    plt.show()

    plt.scatter(x=all_possible_points_x, y=all_possible_points_y)
    plt.xlabel('ECOST')
    plt.ylabel('EGMV')
    plt.show()
    dump([all_possible_points_x, all_possible_points_y], "./single_user_policies_points_{}.pkl".format(seed))


if __name__ == "__main__":
    test_env(seed=1, budget=12000)
