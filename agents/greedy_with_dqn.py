import os
import sys
sys.path.append("../")
from agents.common.common_dqn import DQN2Net_interface
from simulation_env.multiuser_env import run_env, eval_env, MultiUserEnv
import argparse
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class GreedyDQN(DQN2Net_interface):
    def __init__(
            self,
            user_num=1000,
            n_actions=10,
            n_features=29,
            init_roi=10.,
            budget=120.,
            use_budget_control=False,
            use_prioritized_experience_replay=False,
            max_trajectory_length=8,
            update_times_per_train=1,
    ):
        super(GreedyDQN, self).__init__(user_num, n_actions, n_features, init_roi, budget,
                                        use_budget_control=use_budget_control,
                                        use_prioritized_experience_replay=use_prioritized_experience_replay,
                                        max_trajectory_length=max_trajectory_length,
                                        update_times_per_train=update_times_per_train)

    def _pick_loss(self):
        self.has_target_net = False
        self.gmv_loss = self.gmv_montecarlo_loss
        self.cost_loss = self.cost_montecarlo_loss
        self.loss = self.montecarlo_loss
        self.priority_values = self.gmv_montecarlo_error + self.cost_montecarlo_error + self.montecarlo_error

    def _pick_td_loss(self):
        self.has_target_net = True
        self.gmv_loss = self.gmv_dqn_loss
        self.cost_loss = self.cost_dqn_loss
        self.loss = self.dqn_loss
        self.priority_values = self.gmv_td0_error + self.cost_td0_error


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
    # train = False
    user_max_request_time = 7
    n_features = 61
    update_times_per_train = 10
    use_budget_control = True
    use_prioritized_replay = False
    agent = GreedyDQN(
        user_num=user_num,
        n_actions=11,
        n_features=n_features,
        init_roi=init_cpr_thr, budget=budget,
        use_budget_control=use_budget_control,
        use_prioritized_experience_replay=use_prioritized_replay,
        max_trajectory_length=user_max_request_time,
        update_times_per_train=update_times_per_train)

    if train:
        run_env(agent=agent,
                user_num=user_num, training_episode=200, training_log_interval=1, test_interval_list=[100, 5],
                # user_num=user_num, training_episode=1000, training_log_interval=1, test_interval_list=[100, 5],
                test_round=1, seed=seed, init_roi_th=init_cpr_thr,
                use_prioritized_replay=use_prioritized_replay,
                budget=budget,
                use_budget_control=use_budget_control,
                user_max_request_time=user_max_request_time
                )
    else:
        eval_env(
            agent,
            "./exp/learning_result/GreedyDQN/train/1000_user/action_n=11/per=0/seed=1/1581679892/best_model/model-235",
            user_num=user_num, seed=seed, test_epoch=10, init_roi_th=init_cpr_thr,
            print_log=False,
            use_prioritized_replay=use_prioritized_replay,
            budget=budget,
            use_budget_control=use_budget_control,
            user_max_request_time=user_max_request_time)
