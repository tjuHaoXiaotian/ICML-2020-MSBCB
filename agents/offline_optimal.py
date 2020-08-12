import copy
import time
import warnings
import sys

sys.path.append("../")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from simulation_env.multiuser_env import MultiUserEnv, enumerate_policies, dump, reload_data, PIDAgent
import argparse

warnings.filterwarnings(action='once')
large = 22
med = 16
small = 12
params = {'axes.titlesize': large,
          'legend.fontsize': med,

          'figure.figsize': (6, 5),
          'axes.labelsize': med,

          'xtick.labelsize': med,
          'ytick.labelsize': med,
          'figure.titlesize': large,

          'font.family': 'serif',
          'font.serif': 'Times New Roman',
          'font.style': 'normal',
          'font.weight': 'normal',

          'font.size': 20,
          }
plt.rcParams.update(params)

sns.set_style('whitegrid')


def enumerate_all_users(seed=1, user_num=1000, budget=12000):
    MultiUserEnv.seed(seed)
    env = MultiUserEnv(budget=budget, init_roi_th=1., render=False, seed=seed, user_num=user_num, item_num=10,
                       user_max_request_time=7)
    policies = enumerate_policies(env.user_max_request_time)
    print("user max request times={}, policy number={}".format(env.user_max_request_time, len(policies)))

    all_user_all_policy_points = {}
    for policy_idx in range(len(policies)):
        current_policy = policies[policy_idx]
        ob, [user_end, end], _info = env.reset()
        user_idx = _info["user_idx"]
        while not end:
            if all_user_all_policy_points.get(user_idx, None) is None:
                all_user_all_policy_points[user_idx] = []

            epoch_gmv, epoch_cost = 0, 0
            while not user_end:
                bid = MultiUserEnv.bid_max if current_policy[env.request_idx] == 1 else 0
                ob, [gmv, cost, reward], [user_end, end], _info = env.step(bid)
                epoch_cost += cost
                epoch_gmv += gmv
            gmv = round(epoch_gmv * 100)
            cost = round(epoch_cost * 100)
            roi = gmv / (cost + 1e-16)
            all_user_all_policy_points[user_idx].append([gmv, cost, roi])

            user_idx = _info["user_idx"]
            user_end = False

        if policy_idx % 20 == 0:
            print("policy", policy_idx)

    env.close()
    dump(all_user_all_policy_points,
         "./exp/offline_enum/all_{}_user_all_policy_points_seed={}_integer.pkl".format(user_num, seed))


def online_myopic_greedy(seed=1, user_num=1000, budget=6000, epoch=100, init_cpr_thr=6., integration=5,
                         convert2integer=True):
    MultiUserEnv.seed(seed)
    env = MultiUserEnv(budget=budget, init_roi_th=1., render=False, seed=seed, user_num=user_num, item_num=10,
                       user_max_request_time=7)

    if convert2integer:
        budget *= 100

    pid_agent = PIDAgent(init_roi=init_cpr_thr, default_alpha=1., budget=budget, integration=integration)

    all_epoch_gmvs, all_epoch_costs, all_epoch_roi_thrs = [], [], []
    for iteration in range(epoch):
        ob, [user_end, end], _info = env.reset()
        item_price = _info["proxy_ad_price"]
        epoch_gmv, epoch_cost = 0, 0
        current_roi_thr = pid_agent.get_roi_threshold()

        plt.clf()
        while not end:

            user_gmv, user_cost = 0, 0
            while not user_end:
                cvr = _info["cvr"]
                bid = cvr * item_price / current_roi_thr
                ob, [gmv, cost, reward], [user_end, end], _info = env.step(bid)
                user_cost += cost
                user_gmv += gmv
            gmv = round(user_gmv * 100)
            cost = round(user_cost * 100)

            epoch_gmv += gmv
            epoch_cost += cost

            user_end = False

        all_epoch_gmvs.append(epoch_gmv)
        all_epoch_costs.append(epoch_cost)
        all_epoch_roi_thrs.append(current_roi_thr)
        pid_agent.record_cost(iteration, epoch_cost)

        gmv_graph = plt.subplot(1, 3, 1)
        gmv_graph.set_title("GMV")
        gmv_graph.set_xlabel('epoch')
        gmv_graph.set_ylabel('overall-gmv')
        gmv_graph.plot(all_epoch_gmvs, label="epoch gmv", linewidth=2, alpha=1, marker='o')

        cost_graph = plt.subplot(1, 3, 2)
        cost_graph.set_title("Cost")
        cost_graph.set_xlabel('epoch')
        cost_graph.set_ylabel('overall-cost')
        cost_graph.plot(all_epoch_costs, label="epoch cost", linewidth=2, alpha=1, marker='o')

        roi_graph = plt.subplot(1, 3, 3)
        roi_graph.set_title("ROI_th")
        roi_graph.set_xlabel('epoch')
        roi_graph.set_ylabel('roi_thr')
        roi_graph.plot(all_epoch_roi_thrs, label="epoch roi_th", linewidth=2, alpha=1, marker='o')
        plt.pause(0.01)

    plt.ioff()
    plt.show()

    validate_indexes = np.asarray(all_epoch_costs) <= budget
    validate_gmvs = np.asarray(all_epoch_gmvs)[validate_indexes]
    validate_costs = np.asarray(all_epoch_costs)[validate_indexes]
    validate_rois = np.asarray(all_epoch_roi_thrs)[validate_indexes]
    optimal_idx = np.argmax(validate_gmvs)
    print("Optimal result of optimal greedy is:", optimal_idx)
    print("GMV={}, Cost={}, roi_thr={}".format(validate_gmvs[optimal_idx] / 100, validate_costs[optimal_idx] / 100,
                                               validate_rois[optimal_idx]))
    dump([validate_gmvs[optimal_idx] / 100, validate_costs[optimal_idx] / 100,
          validate_rois[optimal_idx], [all_epoch_gmvs, all_epoch_costs, all_epoch_roi_thrs]],
         "./exp/offline_enum/{}_users_myopic_greedy_integer.pkl".format(user_num))


def cal_greedy_with_maximum_CPR(filepath, budget=6000, first_run=False, convert2integer=True):
    if convert2integer:
        budget *= 100
    all_user_all_policy_points = reload_data(filepath)
    all_user_gmv_cost_rois = []
    sum_cost, sum_gmv = 0, 0
    for user_id, all_policy_values in all_user_all_policy_points.items():
        # print(user_id, all_policy_values)
        all_policy_values = np.asarray(all_policy_values)
        argmax_idx = np.argmax(all_policy_values[:, -1])
        gmv, cost, roi = all_policy_values[argmax_idx]

        all_user_gmv_cost_rois.append([gmv, cost, roi])
        sum_cost += cost
        sum_gmv += gmv
    all_user_gmv_cost_rois = np.asarray(all_user_gmv_cost_rois)
    print("sum-gmv={}, sum-cost={}, min-cost={}, max_cost={}, min-roi={}, max_roi={}".format(sum_gmv / 100,
                                                                                             sum_cost / 100,
                                                                                             np.min(
                                                                                                 all_user_gmv_cost_rois[
                                                                                                 :, 1]) / 100, np.max(
            all_user_gmv_cost_rois[:, 1]) / 100,
                                                                                             np.min(
                                                                                                 all_user_gmv_cost_rois[
                                                                                                 :, -1]), np.max(
            all_user_gmv_cost_rois[:, -1])))
    if first_run:
        return
    sorted_all_user_gmv_cost_rois = sorted(all_user_gmv_cost_rois, key=lambda x: x[2], reverse=True)

    cumulative_gmv, cumulative_cost, roi_thr = 0, 0, 0
    for user_triple in sorted_all_user_gmv_cost_rois:
        if cumulative_cost + user_triple[1] > budget:
            break
        else:
            cumulative_gmv += user_triple[0]
            cumulative_cost += user_triple[1]
            roi_thr = user_triple[2]

    print("GMV={}, Cost={}, roi_thr={}".format(cumulative_gmv / 100, cumulative_cost / 100, roi_thr))
    dump([cumulative_gmv / 100, cumulative_cost / 100, roi_thr, sorted_all_user_gmv_cost_rois],
         "./exp/offline_enum/{}_users_greedy_with_maximum_CPR_integer.pkl".format(len(all_user_all_policy_points)))


def cal_greedy_with_optimal_objective_function(filepath, budget=6000, init_cpr_thr=6., epoch=100, integration=5,
                                               convert2integer=True):
    def __get_optimal_user_policies_for_current_roi_thr__(all_user_all_policy_points, roi_thr):

        sum_cost, sum_gmv = 0, 0
        for user_id, all_policy_values in all_user_all_policy_points.items():
            all_policy_values = np.asarray(all_policy_values)
            argmax_idx = np.argmax(
                all_policy_values[:, 0] - roi_thr * all_policy_values[:, 1])
            gmv, cost, roi = all_policy_values[argmax_idx]
            if roi >= roi_thr:
                sum_cost += cost
                sum_gmv += gmv
        return sum_gmv, sum_cost

    if convert2integer:
        budget *= 100

    all_user_all_policy_points = reload_data(filepath)
    pid_agent = PIDAgent(init_roi=init_cpr_thr, default_alpha=1., budget=budget, integration=integration)
    all_epoch_gmvs, all_epoch_costs, all_epoch_roi_thrs = [], [], []
    for iteration in range(epoch):
        plt.clf()

        current_roi_thr = pid_agent.get_roi_threshold()
        sum_gmv, sum_cost = __get_optimal_user_policies_for_current_roi_thr__(all_user_all_policy_points,
                                                                              current_roi_thr)
        all_epoch_gmvs.append(sum_gmv)
        all_epoch_costs.append(sum_cost)
        all_epoch_roi_thrs.append(current_roi_thr)

        pid_agent.record_cost(iteration, sum_cost)
        print("epoch {}:".format(iteration), sum_gmv, sum_cost, current_roi_thr)

        gmv_graph = plt.subplot(1, 3, 1)
        gmv_graph.set_title("GMV")
        gmv_graph.set_xlabel('epoch')
        gmv_graph.set_ylabel('overall-gmv')
        gmv_graph.plot(all_epoch_gmvs, label="epoch gmv", linewidth=2, alpha=1, marker='o')

        cost_graph = plt.subplot(1, 3, 2)
        cost_graph.set_title("Cost")
        cost_graph.set_xlabel('epoch')
        cost_graph.set_ylabel('overall-cost')
        cost_graph.plot(all_epoch_costs, label="epoch cost", linewidth=2, alpha=1, marker='o')

        roi_graph = plt.subplot(1, 3, 3)
        roi_graph.set_title("ROI_th")
        roi_graph.set_xlabel('epoch')
        roi_graph.set_ylabel('roi_thr')
        roi_graph.plot(all_epoch_roi_thrs, label="epoch roi_th", linewidth=2, alpha=1, marker='o')
        plt.pause(0.01)

    plt.ioff()
    plt.show()

    validate_indexes = np.asarray(all_epoch_costs) <= budget
    validate_gmvs = np.asarray(all_epoch_gmvs)[validate_indexes]
    validate_costs = np.asarray(all_epoch_costs)[validate_indexes]
    validate_rois = np.asarray(all_epoch_roi_thrs)[validate_indexes]
    optimal_idx = np.argmax(validate_gmvs)
    print("Optimal result of our method is:", optimal_idx)
    print("GMV={}, Cost={}, roi_thr={}, return={}".format(validate_gmvs[optimal_idx] / 100,
                                                          validate_costs[optimal_idx] / 100,
                                                          validate_rois[optimal_idx],
                                                          validate_gmvs[optimal_idx] / 100 - validate_rois[
                                                              optimal_idx] *
                                                          validate_costs[optimal_idx] / 100))
    dump([validate_gmvs[optimal_idx] / 100, validate_costs[optimal_idx] / 100,
          validate_rois[optimal_idx], [all_epoch_gmvs, all_epoch_costs, all_epoch_roi_thrs]],
         "./exp/offline_enum/{}_users_greedy_with_optimal_objective_function_integer.pkl".format(
             len(all_user_all_policy_points)))


def dynamic_programming(original_enumerate_data_path, user_num=1000, max_request_time=8, budget=6000,
                        convert2integer=True):
    start_time = time.time()
    all_user_all_policy_points = reload_data(original_enumerate_data_path)

    column_num = budget
    if convert2integer:
        column_num = budget * 100

    policy_num = 2 ** max_request_time
    column_num += 1
    row_num = user_num * policy_num + 1
    print("[{}, {}]".format(row_num, column_num))
    dp_package_table = np.zeros([row_num, column_num], dtype=np.float32)
    print("DP table has been initialized...")

    for row_idx in range(1, row_num):
        user_id = (row_idx - 1) // policy_num
        policy_id = (row_idx - 1) % policy_num
        triple = all_user_all_policy_points[user_id][policy_id]
        value, volume = int(triple[0]), int(triple[1])
        for column_idx in range(1, column_num):

            if volume > column_idx:
                dp_package_table[row_idx, column_idx] = dp_package_table[row_idx - 1, column_idx]
            else:
                dp_package_table[row_idx, column_idx] = max(
                    dp_package_table[row_idx - 1, column_idx],
                    dp_package_table[row_idx - (policy_id + 1), column_idx - volume] + value

                )
        if row_idx % 100 == 0:
            print(row_idx)
    end_time = time.time()
    print("The optimal value is {}, cost {} seconds...".format(dp_package_table[row_num - 1, column_num - 1],
                                                               end_time - start_time))
    dump(dp_package_table, original_enumerate_data_path.replace(".pkl", "_dp_table.pkl"))

    print("Begin dump optimal result: ")
    selected_user_and_policy = []
    column_idx = column_num - 1
    row_idx = row_num - 1
    while row_idx != 0:
        if dp_package_table[row_idx][column_idx] != dp_package_table[row_idx - 1][column_idx]:
            user_id = (row_idx - 1) // policy_num
            policy_id = (row_idx - 1) % policy_num
            triple = all_user_all_policy_points[user_id][policy_id]
            value, volume = int(triple[0]), int(triple[1])
            selected_user_and_policy.append([user_id, policy_id, value, volume])
            column_idx -= volume
            row_idx -= (policy_id + 1)
        else:
            row_idx -= 1
    selected_user_and_policy = np.asarray(selected_user_and_policy, dtype=np.int32)
    print("selected user and policy: ", selected_user_and_policy[::-1])
    gmv = np.sum(selected_user_and_policy[:, 2])
    cost = np.sum(selected_user_and_policy[:, 3])
    print("sum of value={}, sum of cost={}, budget={}".format(gmv / 100, cost / 100, column_num - 1))
    dump([selected_user_and_policy, gmv, cost], original_enumerate_data_path.replace(".pkl", "_dp_result.pkl"))


def dynamic_programming_reduce_space(original_enumerate_data_path, user_num=1000, seed=1, max_request_time=8,
                                     budget=6000,
                                     convert2integer=True, continue_from_previous_data=False, previous_path=None):
    start_time = time.time()
    all_user_all_policy_points = reload_data(original_enumerate_data_path)
    column_num = budget
    if convert2integer:
        column_num = budget * 100

    policy_num = 2 ** max_request_time
    column_num += 1
    row_num = user_num * policy_num
    print("[{}, {}]".format(row_num, column_num))

    if continue_from_previous_data:
        assert previous_path is not None
        start_idx, previous_user_last_policy, current_user_previous_and_current_policies = reload_data(previous_path)
        start_idx += 1
    else:
        start_idx = 0
        previous_user_last_policy = np.zeros(column_num, dtype=np.int32)
        current_user_previous_and_current_policies = np.zeros(column_num, dtype=np.int32)
    for row_idx in range(start_idx, row_num):
        user_id = row_idx // policy_num
        policy_id = row_idx % policy_num
        triple = all_user_all_policy_points[user_id][policy_id]
        value, volume = int(triple[0]), int(triple[1])
        for budget_idx in range(column_num - 1, 0, -1):

            if volume <= budget_idx:
                current_user_previous_and_current_policies[budget_idx] = max(
                    current_user_previous_and_current_policies[budget_idx],
                    previous_user_last_policy[budget_idx - volume] + value

                )
        if policy_id == policy_num - 1:
            previous_user_last_policy = copy.deepcopy(current_user_previous_and_current_policies)

        if row_idx % 1000 == 0:
            print(row_idx, "| {} minutes.".format(round((time.time() - start_time) / 60, 2)))
            dump([row_idx, previous_user_last_policy, current_user_previous_and_current_policies],
                 "./exp/offline_enum/DP/{}/seed={}/dp-table-{}.pkl".format(user_num, seed, row_idx))
    end_time = time.time()
    print(
        "The optimal value is {}, cost {} seconds...".format(
            current_user_previous_and_current_policies[column_num - 1] / 100,
            end_time - start_time))


def analyze_cvr_over_estimation(path_train, path_ground_truth, epoch_num):
    train_all_epoch_cvr_over_estimation_gmv_summary = reload_data(path_train)
    true_all_epoch_cvr_over_estimation_gmv_summary = reload_data(path_ground_truth)
    print(len(train_all_epoch_cvr_over_estimation_gmv_summary))
    print(len(true_all_epoch_cvr_over_estimation_gmv_summary))

    train_cvr_over_estimation_gmv_summary = train_all_epoch_cvr_over_estimation_gmv_summary[epoch_num]
    true_cvr_over_estimation_gmv_summary = true_all_epoch_cvr_over_estimation_gmv_summary[epoch_num]
    mask, gmvs = train_cvr_over_estimation_gmv_summary
    mask_true, true_gmvs = true_cvr_over_estimation_gmv_summary

    over_estimate_gmv = np.sum(mask * gmvs - mask * true_gmvs)
    under_estimate_gmv = np.sum((1 - mask) * gmvs - (1 - mask) * true_gmvs)
    print(over_estimate_gmv, under_estimate_gmv, over_estimate_gmv + under_estimate_gmv)

    exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A description of what the program does.")
    parser.add_argument('--seed', default=1, type=int, help='the seed used in the program.')
    parser.add_argument('--user_num', default=10000, type=int, help='the user number of the simulation env.')
    parser.add_argument('--budget', default=12000., type=float, help='the advertising budget of an advertiser.')
    parser.add_argument('--init_cpr_thr', default=6., type=float, help='the init roi_thr.')
    args = parser.parse_args()
    seed = args.seed
    user_num = args.user_num
    # init_cpr_thr = 3.3849124829058534
    init_cpr_thr = args.init_cpr_thr
    budget = args.budget

    # enumerate all offline policies
    enumerate_all_users(seed=seed, user_num=user_num, budget=budget)

    # cal_greedy_with_maximum_CPR("./exp/offline_enum/all_{}_user_all_policy_points_seed=1_integer.pkl".format(user_num),
    #                             budget=budget, first_run=True, convert2integer=True)

    # contextual bandit
    online_myopic_greedy(seed=seed, user_num=user_num, budget=budget, epoch=50, init_cpr_thr=init_cpr_thr,
                         integration=5,  convert2integer=True)

    # greedy with maximized CPR
    cal_greedy_with_maximum_CPR("./exp/offline_enum/all_{}_user_all_policy_points_seed=1_integer.pkl".format(user_num),
                                budget=budget, first_run=False, convert2integer=True)

    # MSBCB(enum)
    cal_greedy_with_optimal_objective_function(
        "./exp/offline_enum/all_{}_user_all_policy_points_seed=1_integer.pkl".format(user_num), budget=budget,
        init_cpr_thr=init_cpr_thr, epoch=100, convert2integer=True)

    exit()
    # offline optimal
    dynamic_programming_reduce_space(
        seed=seed,
        original_enumerate_data_path="./exp/offline_enum/all_{}_user_all_policy_points_seed=1_integer.pkl".format(
            user_num),
        user_num=user_num, max_request_time=7, budget=budget, convert2integer=True, continue_from_previous_data=False,
        previous_path=None)

    """ ENV
    10000 user, request length=7
    sum-gmv=84815.21, sum-cost=12955.97, min-cost=3.49, max_cost=80.19, min-roi=0.002881844380403458, max_roi=23.362725450901802 （对bid做扰动）
    # ************************ Myopic Greedy **************************    
    # GMV=75138.40, Cost=11994.73, roi_thr=2.7647406575960796 
    # *****************************************************************
    # ********* Greedy with maximized ROI ***************************** 
    # GMV=83880.76, Cost=11993.47, roi_thr=1.9109730848861284
    # *****************************************************************
    # ************************* Our MSBCB ***************************** 
    # GMV=89251.77, Cost=11988.36, roi_thr=3.3849124829058534, return=48672.22058643078
    # *****************************************************************
    # ************************ Manual Bid **************************    
    # GMV=33838.27999000624, Cost=11995.100023834035, Reward=21843.17990856804
    # *****************************************************************
    # ************************* DP Optimal ****************************
    # GMV=89291.11, Cost=11999.23
    # *****************************************************************
    """