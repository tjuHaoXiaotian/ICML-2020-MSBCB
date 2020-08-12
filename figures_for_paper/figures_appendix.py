import _pickle as cPickle
import math
import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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

          'font.size': 22,
          }
plt.rcParams.update(params)

sns.set_style('whitegrid')


def reload_data(path):
    with open(path, 'rb') as f:
        return cPickle.load(f, encoding="bytes")


def show_single_user_points(path):
    all_possible_points_x, all_possible_points_y = reload_data(path)
    plt.scatter(x=all_possible_points_x, y=all_possible_points_y, label="diff policies", )

    plt.xlabel('expected-cost')
    plt.ylabel('expected-gmv')

    plt.show()


def show_greedy_with_optimal_objective_function_cmp_with_greedy_with_maximum_roi(myopic_greedy_path, maximum_roi_path,
                                                                                 objective_func_path):
    myopic_greedy_gmv, myopic_greedy_cost, myopic_greedy_roi_thr, [myopic_greedy_gmvs, myopic_greedy_costs,
                                                                   myopic_greedy_roi_thrs] = reload_data(
        myopic_greedy_path)

    greedy_with_maximized_roi_gmv, greedy_with_maximized_roi_cost, greedy_with_maximized_roi_thr, sorted_all_user_gmv_cost_rois = reload_data(
        maximum_roi_path)

    msbcb_gmv, msbcb_cost, msbcb_roi, [msbcb_all_epoch_gmvs, msbcb_all_epoch_costs,
                                       msbcb_all_epoch_roi_thrs] = reload_data(objective_func_path)

    gmv_graph = plt.subplot(1, 3, 1)
    gmv_graph.set_title("GMV")
    gmv_graph.set_xlabel('epoch')
    gmv_graph.set_ylabel('overall-gmv')

    gmv_graph.plot(np.asarray(msbcb_all_epoch_gmvs) / 100, label="msbcb (enum)", linewidth=2, alpha=1, )

    gmv_graph.plot([76961.33] * len(myopic_greedy_gmvs), label="DP (offline optimal)",
                   linewidth=2, alpha=1)

    gmv_graph.legend()

    cost_graph = plt.subplot(1, 3, 2)
    cost_graph.set_title("Cost")
    cost_graph.set_xlabel('epoch')
    cost_graph.set_ylabel('overall-cost')

    cost_graph.plot(np.asarray(msbcb_all_epoch_costs) / 100, label="msbcb (enum)", linewidth=2,
                    alpha=1, )
    cost_graph.plot([6000] * len(myopic_greedy_gmvs), label="DP (offline optimal)",

                    linewidth=2, alpha=1)
    cost_graph.legend()

    roi_graph = plt.subplot(1, 3, 3)
    roi_graph.set_title("ROI_th")
    roi_graph.set_xlabel('epoch')
    roi_graph.set_ylabel('roi_thr')

    roi_graph.plot(msbcb_all_epoch_roi_thrs, label="msbcb (enum)", linewidth=2, alpha=1, )

    roi_graph.legend()

    plt.show()


def show_optimal_point_of_a_given_user(user_idx_list, path, roi_thr, fixed_policy_path=None, learned_roi_thr=None):
    all_user_all_policy_points = reload_data(path)
    if fixed_policy_path is not None:
        all_user_fixed_policy_points = reload_data(fixed_policy_path)

    def f_line(xs, roi_thr):
        return roi_thr * xs

    column_num = 3
    max_row_num = 2
    row_num = len(user_idx_list) // column_num
    rest_column = len(user_idx_list) % column_num
    if rest_column != 0:
        row_num += 1
    row_num = min(row_num, max_row_num)
    print("user_idx_list:", user_idx_list)
    for user_idx in user_idx_list:
        all_policy_values = all_user_all_policy_points[user_idx]
        all_policy_values = np.asarray(all_policy_values) / 100
        argmax_idx = np.argmax(
            all_policy_values[:, 0] - roi_thr * all_policy_values[:, 1])

        figure_idx = user_idx + 1 - user_idx_list[0]

        ax = plt.subplot(max_row_num, column_num, figure_idx)

        x_1 = all_policy_values[:, 1][argmax_idx]
        y_1 = all_policy_values[:, 0][argmax_idx]
        if fixed_policy_path is not None:
            policy_gmv, policy_cost, roi, executed_actions = all_user_fixed_policy_points[user_idx]

            plt.scatter(x=[policy_cost / 100], y=[policy_gmv / 100], label="Learned Policy", s=100)
            plt.scatter(x=[policy_cost / 100], y=[policy_gmv / 100], s=5, c="black")

        x_2 = policy_cost / 100
        y_2 = policy_gmv / 100

        plt.plot(all_policy_values[:, 1], f_line(all_policy_values[:, 1], roi_thr), label="$\mathrm{{CPR}^*_{thr}}$",
                 color="red")

        if user_idx == 7:
            plt.xlabel('Consumer #1')
            plt.ylabel('Consumer #1')
        elif user_idx == 17:
            plt.xlabel('Consumer #2')
            plt.ylabel('Consumer #2')
        else:
            plt.xlabel('Consumer #3')
            plt.ylabel('Consumer #3')
        plt.legend(frameon=True)

        if figure_idx / column_num >= max_row_num:
            break

        k = roi_thr[0]

        if (y_1 / (x_1 + 1e-6) >= k and y_2 / (x_2 + 1e-6) <= k) or (
                y_1 / (x_1 + 1e-6) <= k and y_2 / (x_2 + 1e-6) >= k):
            print("Wired! Optimal ({},{}), Learned ({},{})".format(x_1, y_1, x_2, y_2))

        distance_learned = math.fabs(k * x_2 - y_2) / math.pow(k * k + 1, 0.5)
        distance_optimal = math.fabs(k * x_1 - y_1) / math.pow(k * k + 1, 0.5)
        optimal_ratio = distance_learned / (distance_optimal + 1e-6)
        if optimal_ratio >= 0.90 and optimal_ratio < 1:
            return 0
        elif optimal_ratio >= 0.80:
            return 1
        elif optimal_ratio >= 0.70:
            return 2
        elif optimal_ratio >= 0.60:
            return 3
        elif optimal_ratio >= 0.50:
            return 4
        elif optimal_ratio >= 0.40:
            return 5
        elif optimal_ratio >= 0.30:
            return 6
        elif optimal_ratio >= 0.20:
            return 7
        elif optimal_ratio >= 0.10:
            return 8
        else:
            return 9


def cal_learned_policy_result(fixed_policy_path, roi_thr, budget, learned_roi_thr=None):
    budget = budget
    all_user_fixed_policy_points = reload_data(fixed_policy_path)
    sum_cost, sum_gmv = 0, 0
    all_gmv_cost_list = []
    for user_id, data in all_user_fixed_policy_points.items():
        policy_gmv, policy_cost, roi, executed_actions = data
        policy_gmv = policy_gmv / 100
        policy_cost = policy_cost / 100

        sum_gmv += policy_gmv
        sum_cost += policy_cost
        all_gmv_cost_list.append([policy_gmv, policy_cost, roi])

    sorted_all_gmv_cost_list = sorted(all_gmv_cost_list, key=lambda x: x[-1], reverse=True)

    cumu_cost, cumu_gmv = 0, 0
    for row in sorted_all_gmv_cost_list:
        if cumu_cost + row[1] > budget:
            break
        else:
            cumu_cost += row[1]
            cumu_gmv += row[0]


if __name__ == "__main__":
    fixed_policy_path = "./agents/multi_user/result/20200206_learn_result/MSBCB/test/1000_user/per=0/seed=1/1580965636/epoch_1_user_values.pkl"
    all_p_path = "H:\Projects\sequential_advertising\simulation_env\\agents\multi_user\deterministic7_8_v4\\all_1000_user_all_policy_points_seed=1_integer.pkl"
    learned_roi_thr = None
    roi_thr = 3.3849124829058534,
    budget = 12000.
    cal_learned_policy_result(fixed_policy_path, roi_thr=roi_thr, budget=budget,
                              learned_roi_thr=learned_roi_thr)

    num_90, num_80, num_70, num_60, num_50, num_40, num_30, num_20, num_10, num_00 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    for i in range(1000):
        num = show_optimal_point_of_a_given_user(user_idx_list=[i],
                                                 path=all_p_path,

                                                 roi_thr=roi_thr,
                                                 fixed_policy_path=fixed_policy_path,
                                                 learned_roi_thr=learned_roi_thr
                                                 )
        if num == 0:
            num_90 += 1
        elif num == 1:
            num_80 += 1
        elif num == 2:
            num_70 += 1
        elif num == 3:
            num_60 += 1
        elif num == 4:
            num_50 += 1
        elif num == 5:
            num_40 += 1
        elif num == 6:
            num_30 += 1
        elif num == 7:
            num_20 += 1
        elif num == 8:
            num_10 += 1
        else:
            num_00 += 1
        print(num_90, num_80, num_70, num_60, num_50, num_40, num_30, num_20, num_10, num_00)
