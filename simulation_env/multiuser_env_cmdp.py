import os
import time

import numpy as np
import tensorflow as tf
from simulation_env.multiuser_env import MultiUserEnv, LearningAgent

from plot_util.tf_log import LogScalar, LogHistogram


class MultiUserCMDPEnv(MultiUserEnv):
    cmdp_lambda = [-0.6, -0.4, -0.2, -0.1, 0, +0.1, +0.2, +0.4, +0.6]
    cmdp_lambda_array = np.asarray(cmdp_lambda)
    cmdp_time_steps = 25
    roi_thr_min = 0.5
    roi_thr_max = 10

    @staticmethod
    def find_closest_alpha(action):
        action_idx = np.argmin(np.abs(MultiUserCMDPEnv.cmdp_lambda_array - action))
        return action_idx

    def init_other(self):

        self.cmdp_time_interval = self.user_num // self.cmdp_time_steps
        print("CMDP update interval:", self.cmdp_time_interval)

    def reset_others(self):
        self.cmdp_roi_th = self.roi_th

        self.cmdp_current_t = 0
        self.cmdp_remaining_opportunities = self.cmdp_time_steps - 1
        self.cmdp_remaining_budget = self.budget
        self.cmdp_cumulative_value = 0
        self.cmdp_cumulative_cost = 0
        self.cmdp_cumulative_roi = 0

        self.cmdp_previous_t_remaining_budget = self.budget

        self.cmdp_current_t_cost = 0
        self.cmdp_current_t_value = 0
        self.cmdp_current_t_W_times = 0
        self.cmdp_current_t_request_num = 0

        self.__update_state__()
        self.cmdp_step_DONE = False
        self.info["cmdp_state"] = self.cmdp_state
        self.info["cmdp_step_done"] = self.cmdp_step_DONE
        self.info["roi_thr"] = self.cmdp_roi_th

    def __update_state__(self):
        onehot_cmdp_current_t = [0] * (self.cmdp_time_steps + 1)
        onehot_cmdp_current_t[self.cmdp_current_t] = 1
        onehot_cmdp_remaining_opportunities = [0] * self.cmdp_time_steps
        onehot_cmdp_remaining_opportunities[self.cmdp_remaining_opportunities] = 1
        self.cmdp_state = np.asarray(

            onehot_cmdp_remaining_opportunities +
            [
                self.cmdp_roi_th,
                self.cmdp_remaining_budget / 12000,
                self.cmdp_cumulative_value / 12000,
                self.cmdp_cumulative_cost / 12000,
                self.cmdp_cumulative_roi
            ] +
            [
                self.cmdp_current_t_cost / 100,
                self.cmdp_current_t_cost / 100 / (self.cmdp_current_t_request_num + 1e-8),
                self.cmdp_current_t_value / 100,
                self.cmdp_current_t_value / 100 / (self.cmdp_current_t_request_num + 1e-8),
                self.cmdp_current_t_value / (self.cmdp_current_t_cost + 1e-8),
                self.cmdp_current_t_W_times / (self.cmdp_current_t_request_num + 1e-8),
                self.cmdp_current_t_cost / self.cmdp_previous_t_remaining_budget,
            ]
        )

    def step_others(self, display, expected_gmv, expected_cost, reward):
        self.cmdp_remaining_budget -= expected_cost
        self.cmdp_cumulative_value += expected_gmv
        self.cmdp_cumulative_cost += expected_cost
        self.cmdp_cumulative_roi = self.cmdp_cumulative_value / (self.cmdp_cumulative_cost + 1e-8)

        self.cmdp_current_t_cost += expected_cost
        self.cmdp_current_t_value += expected_gmv
        self.cmdp_current_t_W_times += display
        self.cmdp_current_t_request_num += 1

        if self.cmdp_remaining_budget < 0:
            self.END = True
        if (self.user_idx > 0 and self.user_idx % self.cmdp_time_interval == 0 and self.request_idx == 0) or self.END:

            self.cmdp_current_t += 1
            self.cmdp_remaining_opportunities -= 1
            self.__update_state__()
            self.cmdp_reward = self.cmdp_current_t_value
            self.cmdp_step_DONE = True
            self.info["cmdp_state"] = self.cmdp_state
            self.info["cmdp_step_done"] = self.cmdp_step_DONE
            self.info["cmdp_reward"] = self.cmdp_reward
            self.info["roi_thr"] = self.cmdp_roi_th

            self.cmdp_previous_t_remaining_budget = self.cmdp_remaining_budget

            self.cmdp_current_t_cost = 0
            self.cmdp_current_t_value = 0
            self.cmdp_current_t_W_times = 0
            self.cmdp_current_t_request_num = 0
        else:
            self.cmdp_step_DONE = False
            self.info["cmdp_step_done"] = self.cmdp_step_DONE
            self.info["roi_thr"] = self.cmdp_roi_th

    def update_roi_thr(self, cmdp_alpha_idx):
        previous_roi_thr = self.cmdp_roi_th
        self.cmdp_roi_th = np.clip(self.cmdp_roi_th * (1 + self.cmdp_lambda[cmdp_alpha_idx]), a_min=self.roi_thr_min,
                                   a_max=self.roi_thr_max)
        executed_action = self.cmdp_roi_th / previous_roi_thr - 1
        action_idx = self.find_closest_alpha(executed_action)

        return action_idx

    def update_roi_thr_continous(self, cmdp_alpha):
        previous_roi_thr = self.cmdp_roi_th
        self.cmdp_roi_th = np.clip(self.cmdp_roi_th * (cmdp_alpha), a_min=self.roi_thr_min,
                                   a_max=self.roi_thr_max)
        executed_action = self.cmdp_roi_th / previous_roi_thr - 1
        action_idx = self.find_closest_alpha(executed_action)

        return action_idx

    def get_roi_threshold(self):
        return self.cmdp_roi_th


class CMDPAgent(LearningAgent):
    def get_cmdp_action(self, sess, obs, is_test=False, other_info=None):
        pass

    def experience_cmdp(self, new_trajectory, other_info=None):
        pass

    def train_cvr(self, sess):
        return False, [0, 0, 0]

    def get_agent_name(self):
        pass


def run_env(agent, budget, user_num=1, training_episode=100, training_log_interval=10, test_interval_list=5,
            test_round=1,
            seed=12345, init_roi_th=1., use_prioritized_replay=False,
            use_budget_control=False, user_max_request_time=8):
    start_time = time.time()
    print("use_budget_control={}, budget={}".format(use_budget_control, budget))

    assert isinstance(agent, CMDPAgent)
    BEST_TEST_RETURN = 0

    env = MultiUserCMDPEnv(init_roi_th=init_roi_th, render=False, user_num=user_num, topic_num=10, item_num=10,
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
        "q_loss", "montecarlo_loss", "q_value", "q_value_true", "q_value_diff",
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

        agent.init_parameters(sess=tf_sess)

        tf_sess.graph.finalize()

        train_returns, train_gmvs, train_costs, train_actions, train_bids = [], [], [], [], []
        test_returns, test_gmvs, test_costs, test_actions, test_bids, test_all_actions, test_gaps_to_second_price = [], [], [], [], [], [], []

        training_ep, testing_ep = 0, 0
        is_test = False

        state, [user_done, done], _info = env.reset()
        cmdp_state = _info["cmdp_state"]

        cmdp_action = agent.get_cmdp_action(tf_sess, cmdp_state, is_test=is_test, other_info=_info)
        if agent.get_agent_name() == "CDDPG-model":

            cmdp_action = env.update_roi_thr_continous(cmdp_action)
        else:
            cmdp_action = env.update_roi_thr(cmdp_action)

        test_times = 0
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

                    test_times += 1
                    if training_ep == training_episode:
                        break

            new_trajectory, cvr_trajectory = [], []
            traj_gmv, traj_cost, traj_reward, traj_action, traj_bid, traj_gap_to_second_price = [], [], [], [], [], []
            episode_cost, episode_gmv = 0, 0
            while not done:
                bid, other_action_info = agent.get_action(tf_sess, state, is_test=is_test, other_info=_info)
                state_tp1, [gmv, cost, reward], [user_done, done], _info = env.step(bid, is_test=is_test)
                if _info["cmdp_step_done"]:
                    cmdp_state_tp1 = _info["cmdp_state"]
                    cmdp_reward = _info["cmdp_reward"]
                    new_trajectory.append([cmdp_state, cmdp_action, cmdp_reward, cmdp_state_tp1, done])

                    cmdp_state = cmdp_state_tp1

                    cmdp_action = agent.get_cmdp_action(tf_sess, cmdp_state, is_test=is_test, other_info=_info)
                    if agent.get_agent_name() == "CDDPG-model":

                        cmdp_action = env.update_roi_thr_continous(cmdp_action)
                    else:
                        cmdp_action = env.update_roi_thr(cmdp_action)

                cvr_trajectory.append([state, _info["cvr"]])
                if user_done:
                    if not is_test:
                        agent.experience(None, {
                            "cvr": cvr_trajectory
                        })
                    cvr_trajectory = []

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

            agent.experience_cmdp(new_trajectory)

            if is_test:

                logger_scalar.log(pid_curves, [env.get_roi_threshold(), episode_cost, episode_gmv],
                                  global_step=training_ep + testing_ep)

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

                trained, [cvr_loss, avg_predicted_cvrs, avg_cvr_targets] = agent.train_cvr(tf_sess)
                if training_ep % training_log_interval == 0 and trained:
                    logger_scalar.log(cvr_curves, [cvr_loss, avg_predicted_cvrs, avg_cvr_targets],
                                      global_step=training_ep)

                cmdp_trained, [loss, montecarlo_loss, q_eval, returns], \
                avg_buffer_return, epsilon = agent.train(tf_sess)
                if training_ep % training_log_interval == 0 and cmdp_trained:
                    logger_scalar.log(training_curves, [
                        loss, montecarlo_loss, q_eval, returns, q_eval - returns,
                        avg_buffer_return, epsilon
                    ], global_step=training_ep)

                if training_ep % 100 == 0:
                    seconds = time.time() - start_time
                    print("{}: losses=[cvr-loss={}, dqn-loss={}], cost {} seconds ({} minutes)...".format(
                        training_ep,
                        cvr_loss,
                        loss,
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
            cmdp_state = _info["cmdp_state"]

            cmdp_action = agent.get_cmdp_action(tf_sess, cmdp_state, is_test=is_test, other_info=_info)
            if agent.get_agent_name() == "CDDPG-model":

                cmdp_action = env.update_roi_thr_continous(cmdp_action)
            else:
                cmdp_action = env.update_roi_thr(cmdp_action)

    env.close()


def test_env(seed=1):
    MultiUserCMDPEnv.seed(1)
    env = MultiUserCMDPEnv(init_roi_th=1., render=False, seed=seed, user_num=1000, item_num=10, user_max_request_time=7,
                           budget=1200000)
    agent = lambda ob: env.action_space.sample()[0]
    all_user_ctr = {}
    for _ in range(1):
        sum_gmv, sum_cost, sum_reward = 0, 0, 0
        epoch_cost = []
        ob, [user_end, end], _info = env.reset()
        while not end:
            if all_user_ctr.get(_info["user_idx"], None) is None:
                all_user_ctr[_info["user_idx"]] = []
            all_user_ctr[_info["user_idx"]].append(_info["cvr"])
            a = agent(ob)

            ob, [gmv, cost, reward], [user_end, end], _info = env.step(MultiUserEnv.bid_max)
            epoch_cost.append(cost)
            sum_gmv += gmv
            sum_cost += cost
            sum_reward += reward
        print("GMV={}, Cost={}, Reward={}".format(sum_gmv, sum_cost, sum_reward))
    env.close()


if __name__ == "__main__":
    test_env()
