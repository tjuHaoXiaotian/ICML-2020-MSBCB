import pandas as pd
import matplotlib.pylab as plt
import numpy as np
import os
import seaborn as sns

large = 22
med = 16
small = 12
line_width = 6
params = {'axes.titlesize': large,
          'legend.fontsize': 25,

          'figure.figsize': (6, 5),
          'axes.labelsize': 25,

          'xtick.labelsize': 25,
          'ytick.labelsize': 25,
          'figure.titlesize': large,

          'font.family': 'serif',
          'font.serif': 'Times New Roman',
          'font.style': 'normal',
          'font.weight': 'normal',

          'font.size': 25,
          }
plt.rcParams.update(params)

sns.set_style('whitegrid')


def return_file_list(data_dir_path, exceptions):
    file_names = []
    files = os.listdir(data_dir_path)
    for file in files:
        is_in_exceptions = False
        for exception in exceptions:
            if file.find(exception) != -1:
                is_in_exceptions = True
                break
        if not is_in_exceptions:
            file_names.append(os.path.join(data_dir_path, file))
    return file_names


def return_data_list(file_list):
    data_list = []
    for file in file_list:
        data = pd.read_csv(file)
        data_step = data["Step"].tolist()
        data_value = data["Value"].tolist()
        data_list.append(data_value)
    return data_step, data_list


def draw_avg_curve_for_method(plt, data_dir_path, method_name, marker, show=False, exceptions=(), color="auto",
                              use_move_avg=False, show_variance=False):
    files = return_file_list(data_dir_path, exceptions)
    data_step, data_list = return_data_list(files)
    data_step = np.asarray(data_step)
    mask = data_step >= 100
    data_array = np.asarray(data_list)
    markevery = 10

    if use_move_avg:
        time_horizon = 10
        mean_values = []
        std_values = []
        for idx in range(len(data_step)):
            data_with_window = data_array[:, max(idx - time_horizon, 0): idx + 1]
            point_mean = data_with_window.mean()
            point_std = data_with_window.std()
            mean_values.append(point_mean)
            std_values.append(point_std)
        mean_values = np.asarray(mean_values)
        std_values = np.asarray(std_values)
    else:
        mean_values = data_array.mean(axis=0)
        std_values = data_array.std(axis=0)

    print(method_name, mean_values[-1], std_values[-1])
    if color != "auto":
        plt.plot(data_step[mask], mean_values[mask], marker=marker, markersize=10, markevery=markevery,
                 label=method_name, linewidth=line_width, color=color[0])
    else:
        plt.plot(data_step[mask], mean_values[mask], marker=marker, markersize=10, markevery=markevery,
                 label=method_name, linewidth=line_width)

    variance_time = 1
    plt.fill_between(data_step[mask], mean_values[mask] - variance_time * std_values[mask],
                     mean_values[mask] + variance_time * std_values[mask], color=color[1], alpha=0.4)
    if show:
        plt.show()
    return data_step, mask


def figure_1():
    name_and_paths = [
        ["MSBCB(RL with action space reduction)", "./main_result/learn/GMV/MSBCB/", [], "auto"],
        ["Greedy+DQN", "./main_result/learn/GMV/DQN/", ["1580044437"], "auto"],
        ["Greedy+DDPG", "./main_result/learn/GMV/DDPG/", [], "auto"],
        ["Greedy+PPO", "./main_result/learn/GMV/PPO/", [], "auto"],

        ["Constrained-DQN", "./main_result/learn/GMV/CDQN/", [], "auto"],
    ]
    for name, data_dir, exceptions, color in name_and_paths:
        data_step, mask = draw_avg_curve_for_method(plt, data_dir, name, False, exceptions, color, use_move_avg=True,
                                                    show_variance=True)

    other_datas = [
        ["MSBCB(enum)", 89251.77, "auto", "auto"],
        ["Offline Optimal(dynamic programming,enum)", 89258.21, "auto", "auto"],
        ["Greedy+Maximized CPR(enum)", 83880.76, "auto", "auto", ],
        ["Myopic Greedy(enum)", 75138.40, "auto", "auto", ],

    ]

    for method_name, value, color, marker in other_datas:
        if color != "auto":
            plt.plot(data_step[mask], np.ones_like(data_step[mask]) * value, label=method_name, linewidth=line_width,
                     color=color, linestyle="--")
            if marker != "auto":
                plt.plot(data_step[mask], np.ones_like(data_step[mask]) * value, label=method_name,
                         linewidth=line_width, color=color, linestyle="-.", marker=marker, markersize=10, )
        else:
            if marker != "auto":
                plt.plot(data_step[mask], np.ones_like(data_step[mask]) * value, label=method_name,
                         linewidth=line_width, linestyle="-.", marker=marker, markersize=10, )
            else:
                plt.plot(data_step[mask], np.ones_like(data_step[mask]) * value, label=method_name,
                         linewidth=line_width, linestyle="-.", )
        print(method_name, value)

    plt.xlabel('Training Episode')
    plt.ylabel('Value')
    plt.legend(loc="lower right", )
    plt.show()


def figure_2():
    name_and_paths = [
        ["MSBCB(RL with action space reduction)", "./main_result/learn/Cost/MSBCB/", [], "auto"],
        ["Greedy+DQN", "./main_result/learn/Cost/DQN/", ["1580044437"], "auto"],
        ["Greedy+DDPG", "./main_result/learn/Cost/DDPG/", [], "auto"],
        ["Greedy+PPO", "./main_result/learn/Cost/PPO/", [], "auto"],

        ["Constrained-DQN", "./main_result/learn/Cost/CDQN/", [], "auto"],
    ]
    for name, data_dir, exceptions, color in name_and_paths:
        data_step, mask = draw_avg_curve_for_method(plt, data_dir, name, False, exceptions, color, use_move_avg=True,
                                                    show_variance=True)

    other_datas = [
        ["MSBCB(enum)", 11988.36, "auto", "auto"],
        ["Offline Optimal(dynamic programming,enum)", 11999.23, "auto", "auto"],
        ["Greedy+Maximized CPR(enum)", 11993.47, "auto", "auto", ],
        ["Myopic Greedy(enum)", 11994.73, "auto", "auto", ],

    ]
    show_other_fixed_data(plt, other_datas, data_step, mask)

    plt.xlabel('Training Episode')
    plt.ylabel('Cost')
    plt.legend(loc="lower right", )
    plt.show()


def show_other_fixed_data(plt, other_datas, data_step, mask):
    for method_name, value, color, marker in other_datas:
        if color != "auto":
            plt.plot(data_step[mask], np.ones_like(data_step[mask]) * value, label=method_name, linewidth=line_width,
                     color=color, linestyle="-.")
            if marker != "auto":
                plt.plot(data_step[mask], np.ones_like(data_step[mask]) * value, label=method_name,
                         linewidth=line_width, color=color, marker=marker, markersize=10, linestyle="-.")
        else:
            if marker != "auto":
                plt.plot(data_step[mask], np.ones_like(data_step[mask]) * value, label=method_name,
                         linewidth=line_width, marker=marker, markersize=10, linestyle="-.")
            else:
                plt.plot(data_step[mask], np.ones_like(data_step[mask]) * value, label=method_name,
                         linewidth=line_width, linestyle="-.")
        print(method_name, value)


def main_sub_fiture_1():
    color_list = [
        [[224 / 255, 78 / 255, 81 / 255], [247 / 255, 209 / 255, 210 / 255]],
        [[128 / 255, 128 / 255, 128 / 255], [204 / 255, 204 / 255, 204 / 255]],
        [[100 / 255, 212 / 255, 84 / 255], [125 / 255, 192 / 255, 164 / 255]],
        [[255 / 255, 136 / 255, 22 / 255], [255 / 255, 231 / 255, 209 / 255]],
        [[208 / 255, 119 / 255, 195 / 255], [246 / 255, 228 / 255, 243 / 255]],
        [[28 / 255, 69 / 255, 135 / 255], [164 / 255, 181 / 255, 207 / 255]],
        [[57 / 255, 139 / 255, 191 / 255], [213 / 255, 231 / 255, 242 / 255]],
        [[255 / 255, 51 / 255, 153 / 255], [255 / 255, 215 / 255, 235 / 255]],
    ]
    name_and_paths = [
        ["MSBCB", "./main_result/learn/GMV/MSBCB/", [], color_list[0], "o"],

        ["Contextual Bandit", "./main_result/learn/GMV/MyopicGreedyTrue/", [], color_list[6], "v"],
    ]
    for name, data_dir, exceptions, color, marker in name_and_paths:
        data_step, mask = draw_avg_curve_for_method(plt, data_dir, name, marker, False, exceptions, color,
                                                    use_move_avg=True,
                                                    show_variance=True)

    other_datas = [

        ["Manual Bid", 38838.27999000624, color_list[2][0], "auto", ],
        ["Offline Optimal", 89291.11, color_list[5][0], "auto", ],

    ]
    show_other_fixed_data(plt, other_datas, data_step, mask)

    plt.xlabel('Training Episode')
    plt.ylabel('Cumulative Value (Revenue)')
    plt.legend(frameon=True, loc='best')

    plt.show()


def main_sub_fiture_2():
    color_list = [
        [[224 / 255, 78 / 255, 81 / 255], [247 / 255, 209 / 255, 210 / 255]],
        [[128 / 255, 128 / 255, 128 / 255], [204 / 255, 204 / 255, 204 / 255]],
        [[100 / 255, 212 / 255, 84 / 255], [125 / 255, 192 / 255, 164 / 255]],
        [[255 / 255, 136 / 255, 22 / 255], [255 / 255, 231 / 255, 209 / 255]],
        [[208 / 255, 119 / 255, 195 / 255], [246 / 255, 228 / 255, 243 / 255]],
        [[28 / 255, 69 / 255, 135 / 255], [164 / 255, 181 / 255, 207 / 255]],
        [[57 / 255, 139 / 255, 191 / 255], [213 / 255, 231 / 255, 242 / 255]],
        [[255 / 255, 51 / 255, 153 / 255], [255 / 255, 215 / 255, 235 / 255]],

    ]
    name_and_paths = [
        ["MSBCB", "./main_result/learn/GMV/MSBCB/", [], color_list[0], "o"],
        ["Greedy + PPO", "./main_result/learn/GMV/PPO/", [], color_list[4], "v"],
        ["Greedy + DDPG", "./main_result/learn/GMV/DDPG/", [], color_list[2], "s"],
        ["Greedy + DQN", "./main_result/learn/GMV/DQN/", ["1580044437"], color_list[3], "p"],
        ["Greedy + maxCPR", "./main_result/learn/GMV/MSBCB-ROI/", ["1580044843"], color_list[5], "D"],
    ]
    for name, data_dir, exceptions, color, marker in name_and_paths:
        data_step, mask = draw_avg_curve_for_method(plt, data_dir, name, marker, False, exceptions, color,
                                                    use_move_avg=True,
                                                    show_variance=True)

    other_datas = [

    ]
    show_other_fixed_data(plt, other_datas, data_step, mask)

    plt.xlabel('Training Episode')
    plt.ylabel('V(G)')
    plt.legend(frameon=True)

    plt.show()


def main_sub_fiture_3():
    color_list = [
        [[224 / 255, 78 / 255, 81 / 255], [247 / 255, 209 / 255, 210 / 255]],
        [[128 / 255, 128 / 255, 128 / 255], [204 / 255, 204 / 255, 204 / 255]],
        [[100 / 255, 212 / 255, 84 / 255], [125 / 255, 192 / 255, 164 / 255]],
        [[255 / 255, 136 / 255, 22 / 255], [255 / 255, 231 / 255, 209 / 255]],
        [[208 / 255, 119 / 255, 195 / 255], [246 / 255, 228 / 255, 243 / 255]],
        [[28 / 255, 69 / 255, 135 / 255], [164 / 255, 181 / 255, 207 / 255]],
        [[57 / 255, 139 / 255, 191 / 255], [213 / 255, 231 / 255, 242 / 255]],
        [[255 / 255, 51 / 255, 153 / 255], [255 / 255, 215 / 255, 235 / 255]],

    ]
    name_and_paths = [
        ["MSBCB", "./main_result/learn/GMV/MSBCB/", [], color_list[0], "o"],
        ["Constrained + DQN", "./main_result/learn/GMV/CDQN/", [], color_list[2], "v"],
        ["Constrained + DDPG", "./main_result/learn/GMV/CDDPG/", [], color_list[3], "s"],
        ["Constrained + PPO", "./main_result/learn/GMV/CPPO/", [], color_list[6], "p"],
    ]
    for name, data_dir, exceptions, color, marker in name_and_paths:
        data_step, mask = draw_avg_curve_for_method(plt, data_dir, name, marker, False, exceptions, color,
                                                    use_move_avg=True,
                                                    show_variance=True)

    other_datas = [

    ]
    show_other_fixed_data(plt, other_datas, data_step, mask)

    plt.xlabel('Training Episode')
    plt.ylabel('V(G)')
    plt.legend(frameon=True)

    plt.show()


def cost_fiture():
    color_list = [
        [[224 / 255, 78 / 255, 81 / 255], [247 / 255, 209 / 255, 210 / 255]],
        [[128 / 255, 128 / 255, 128 / 255], [204 / 255, 204 / 255, 204 / 255]],
        [[100 / 255, 212 / 255, 84 / 255], [125 / 255, 192 / 255, 164 / 255]],
        [[255 / 255, 136 / 255, 22 / 255], [255 / 255, 231 / 255, 209 / 255]],
        [[208 / 255, 119 / 255, 195 / 255], [246 / 255, 228 / 255, 243 / 255]],
        [[28 / 255, 69 / 255, 135 / 255], [164 / 255, 181 / 255, 207 / 255]],
        [[57 / 255, 139 / 255, 191 / 255], [213 / 255, 231 / 255, 242 / 255]],
        [[255 / 255, 51 / 255, 153 / 255], [255 / 255, 215 / 255, 235 / 255]],

    ]
    name_and_paths = [
        ["Contextual Bandit", "./main_result/learn/Cost/MyopicGreedyTrue/", [], color_list[0], "s"],
        ["Constrained + DQN", "./main_result/learn/Cost/CDQN/", [], color_list[7], "p"],
        ["Constrained + DDPG", "./main_result/learn/Cost/CDDPG/", [], color_list[6], "D"],
        ["Constrained + PPO", "./main_result/learn/Cost/CPPO/", [], color_list[5], "o"],
        ["Greedy + DQN", "./main_result/learn/Cost/DQN/", ["1580044437"], color_list[2], "p"],
        ["Greedy + DDPG", "./main_result/learn/Cost/DDPG/", [], color_list[3], "s"],
        ["Greedy + PPO", "./main_result/learn/Cost/PPO/", [], color_list[4], "v"],
        ["Greedy + maxCPR", "./main_result/learn/Cost/MSBCB-ROI/", ["1580044781"], color_list[1], "D"],
        ["MSBCB", "./main_result/learn/Cost/MSBCB/", [], color_list[0], "o"],
    ]
    for name, data_dir, exceptions, color, marker in name_and_paths:
        data_step, mask = draw_avg_curve_for_method(plt, data_dir, name, marker, False, exceptions, color,
                                                    use_move_avg=True,
                                                    show_variance=True)

    other_datas = [

        ["Manual Bid", 12000, color_list[2][0], "auto", ],
        ["Offline Optimal", 11999.23, color_list[5][0], "auto", ],

    ]
    show_other_fixed_data(plt, other_datas, data_step, mask)

    plt.xlabel('Training Episode')
    plt.ylabel('V(C)')
    plt.legend(frameon=True, fontsize=20)

    plt.show()


def roi_figure():
    color_list = [
        [[224 / 255, 78 / 255, 81 / 255], [247 / 255, 209 / 255, 210 / 255]],
        [[128 / 255, 128 / 255, 128 / 255], [204 / 255, 204 / 255, 204 / 255]],
        [[100 / 255, 212 / 255, 84 / 255], [125 / 255, 192 / 255, 164 / 255]],
        [[255 / 255, 136 / 255, 22 / 255], [255 / 255, 231 / 255, 209 / 255]],
        [[208 / 255, 119 / 255, 195 / 255], [246 / 255, 228 / 255, 243 / 255]],
        [[28 / 255, 69 / 255, 135 / 255], [164 / 255, 181 / 255, 207 / 255]],
        [[57 / 255, 139 / 255, 191 / 255], [213 / 255, 231 / 255, 242 / 255]],
        [[255 / 255, 51 / 255, 153 / 255], [255 / 255, 215 / 255, 235 / 255]],

    ]
    name_and_paths = [
        ["MSBCB", "./main_result/learn/ROI/MSBCB/", [], color_list[0], "o"],
        ["Greedy + PPO", "./main_result/learn/ROI/PPO/", [], color_list[4], "v"],
        ["Greedy + DDPG", "./main_result/learn/ROI/DDPG/", [], color_list[2], "s"],
        ["Greedy + DQN", "./main_result/learn/ROI/DQN/", [], color_list[3], "p"],
    ]
    for name, data_dir, exceptions, color, marker in name_and_paths:
        data_step, mask = draw_avg_curve_for_method(plt, data_dir, name, marker, False, exceptions, color,
                                                    use_move_avg=True,
                                                    show_variance=False)

    other_datas = [
        ["$\mathrm{{CPR}^*_{thr}}$", 3.16, "auto", "auto", ],
    ]
    show_other_fixed_data(plt, other_datas, data_step, mask)

    plt.xlabel('Training Episode')
    plt.ylabel('$\mathrm{{CPR}_{thr}}$')
    plt.ylim((0, 5))
    plt.legend(frameon=True)

    plt.show()


def interpolated_intercept(x, y1, y2):
    """Find the intercept of two curves, given by the same x data"""

    def intercept(point1, point2, point3, point4):
        """find the intersection between two lines
        the first line is defined by the line between point1 and point2
        the first line is defined by the line between point3 and point4
        each point is an (x,y) tuple.

        So, for example, you can find the intersection between
        intercept((0,0), (1,1), (0,1), (1,0)) = (0.5, 0.5)

        Returns: the intercept, in (x,y) format
        """

        def line(p1, p2):
            A = (p1[1] - p2[1])
            B = (p2[0] - p1[0])
            C = (p1[0] * p2[1] - p2[0] * p1[1])
            return A, B, -C

        def intersection(L1, L2):
            D = L1[0] * L2[1] - L1[1] * L2[0]
            Dx = L1[2] * L2[1] - L1[1] * L2[2]
            Dy = L1[0] * L2[2] - L1[2] * L2[0]

            x = Dx / D
            y = Dy / D
            return x, y

        L1 = line([point1[0], point1[1]], [point2[0], point2[1]])
        L2 = line([point3[0], point3[1]], [point4[0], point4[1]])

        R = intersection(L1, L2)

        return R

    idx = np.argwhere(np.diff(np.sign(y1 - y2)) != 0)
    xc, yc = intercept((x[idx], y1[idx]), ((x[idx + 1], y1[idx + 1])), ((x[idx], y2[idx])), ((x[idx + 1], y2[idx + 1])))
    return xc, yc


def mat_plot_intersec():
    color_list = [

        [[28 / 255, 69 / 255, 135 / 255], [164 / 255, 181 / 255, 207 / 255]],
        [[100 / 255, 212 / 255, 84 / 255], [225 / 255, 247 / 255, 221 / 255]],
        [[255 / 255, 136 / 255, 22 / 255], [255 / 255, 231 / 255, 209 / 255]],
        [[208 / 255, 119 / 255, 195 / 255], [246 / 255, 228 / 255, 243 / 255]],

        [[255 / 255, 51 / 255, 153 / 255], [255 / 255, 215 / 255, 235 / 255]],
        [[100 / 255, 139 / 255, 89 / 255], [167 / 255, 194 / 255, 157 / 255]],
        [[224 / 255, 78 / 255, 81 / 255], [247 / 255, 209 / 255, 210 / 255]],
        [[57 / 255, 139 / 255, 191 / 255], [213 / 255, 231 / 255, 242 / 255]],
        [[128 / 255, 128 / 255, 128 / 255], [204 / 255, 204 / 255, 204 / 255]],

    ]
    name_and_paths = [
        ["./main_result/learn/GMV/MSBCB/", []],
        ["./main_result/learn/GMV/DQN/", []],
        ["./main_result/learn/GMV/DDPG/", []],
        ["./main_result/learn/GMV/PPO/", []],
    ]
    file_list = []
    for data_dir, exceptions in name_and_paths:
        files = return_file_list(data_dir, exceptions)

        file_list.append(files)
    print(file_list)
    value_list = [[] for i in range(len(file_list))]
    i = 0
    for files in file_list:
        for file in files:
            data = pd.read_csv(file)

            data_value = np.array(data["Value"]).tolist()
            value_list[i].append(data_value)
        i = i + 1

    label_list = [
        "MSBCB",
        "Greedy + DQN",
        "Greedy + DDPG",
        "Greedy + PPO",

    ]

    step = np.array([i * 5.6 for i in range(len(value_list[0][0]))])

    y1 = np.array([60000 for i in range(len(value_list[0][0]))])
    y2 = np.array([65000 for i in range(len(value_list[0][0]))])
    y3 = np.array([70000 for i in range(len(value_list[0][0]))])
    y4 = np.array([75000 for i in range(len(value_list[0][0]))])
    y5 = np.array([80000 for i in range(len(value_list[0][0]))])
    y6 = np.array([85000 for i in range(len(value_list[0][0]))])

    plt.plot(step, y1, marker='8', mec='none', ms=2, lw=1, label='60000 baseline', color=color_list[4][0])
    plt.plot(step, y2, marker='8', mec='none', ms=2, lw=1, label='65000 baseline', color=color_list[5][0])
    plt.plot(step, y3, marker='8', mec='none', ms=2, lw=1, label='70000 baseline', color=color_list[6][0])
    plt.plot(step, y4, marker='8', mec='none', ms=2, lw=1, label='75000 baseline', color=color_list[7][0])
    plt.plot(step, y5, marker='8', mec='none', ms=2, lw=1, label='80000 baseline', color=color_list[8][0])
    plt.plot(step, y6, marker='8', mec='none', ms=2, lw=1, label='85000 baseline', color=color_list[3][0])

    for i in range(len(value_list)):
        values_array = np.array(value_list[i])
        time_horizon = 10
        mean_values = []
        std_values = []
        for idx in range(len(step)):
            data_with_window = values_array[:, max(idx - time_horizon, 0): idx + 1]
            point_mean = data_with_window.mean()
            point_std = data_with_window.std()
            mean_values.append(point_mean)
            std_values.append(point_std)

        values_mean = np.asarray(mean_values)
        values_std = np.asarray(std_values)

        plt.plot(step, values_mean, color=color_list[i][0], label=label_list[i], linewidth=line_width)

        buffer = 512 * 10
        xc1, yc1 = interpolated_intercept(step, values_mean, y1)
        print(
            "{} use {} steps  and {} trajectories to reach {}".format(label_list[i], xc1[0][0], int(xc1[0][0]) * buffer,
                                                                      yc1[0]))
        plt.plot(xc1, yc1, 'co', ms=10, color='red')

        xc2, yc2 = interpolated_intercept(step, values_mean, y2)
        print(
            "{} use {} steps  and {} trajectories to reach {}".format(label_list[i], xc2[0][0], int(xc2[0][0]) * buffer,
                                                                      yc2[0]))
        plt.plot(xc2, yc2, 'co', ms=10, color='red', linewidth=line_width)

        xc3, yc3 = interpolated_intercept(step, values_mean, y3)
        print(
            "{} use {} steps  and {} trajectories to reach {}".format(label_list[i], xc3[0][0], int(xc3[0][0]) * buffer,
                                                                      yc3[0]))
        plt.plot(xc3, yc3, 'co', ms=10, color='red', linewidth=line_width)

        xc4, yc4 = interpolated_intercept(step, values_mean, y4)
        print(
            "{} use {} steps  and {} trajectories to reach {}".format(label_list[i], xc4[0][0], int(xc4[0][0]) * buffer,
                                                                      yc4[0]))
        plt.plot(xc4, yc4, 'co', ms=10, color='red', linewidth=line_width)

        if label_list[i] != 'Greedy + PPO':
            xc5, yc5 = interpolated_intercept(step, values_mean, y5)
            print("{} use {} steps  and {} trajectories to reach {}".format(label_list[i], xc5[0][0],
                                                                            int(xc5[0][0]) * buffer, yc5[0]))
            plt.plot(xc5[0], yc5[0], 'co', ms=10, color='red', linewidth=line_width)

        if label_list[i] == 'MSBCB':
            xc6, yc6 = interpolated_intercept(step, values_mean, y6)
            print("{} use {} steps  and {} trajectories to reach {}".format(label_list[i], xc6[0][0],
                                                                            int(xc6[0][0]) * buffer, yc6[0]))
            plt.plot(xc6, yc6, 'co', ms=10, color='red', linewidth=line_width)

    plt.legend(frameon=True, fontsize=25, numpoints=1, loc='lower right', ncol=2)
    plt.xlabel('Training Episode')
    plt.ylabel('V(G)')

    plt.show()


def second_bid_figure():
    color_list = [
        [[224 / 255, 78 / 255, 81 / 255], [247 / 255, 209 / 255, 210 / 255]],
        [[128 / 255, 128 / 255, 128 / 255], [204 / 255, 204 / 255, 204 / 255]],
        [[100 / 255, 212 / 255, 84 / 255], [125 / 255, 192 / 255, 164 / 255]],
        [[255 / 255, 136 / 255, 22 / 255], [255 / 255, 231 / 255, 209 / 255]],
        [[208 / 255, 119 / 255, 195 / 255], [246 / 255, 228 / 255, 243 / 255]],
        [[28 / 255, 69 / 255, 135 / 255], [164 / 255, 181 / 255, 207 / 255]],
        [[57 / 255, 139 / 255, 191 / 255], [213 / 255, 231 / 255, 242 / 255]],
        [[255 / 255, 51 / 255, 153 / 255], [255 / 255, 215 / 255, 235 / 255]],

    ]
    name_and_paths = [
        ["MSBCB", "./main_result/learn/GapToSecondPrice/MSBCB/", [], color_list[0], "o"],
        ["Greedy + PPO", "./main_result/learn/GapToSecondPrice/PPO/", [], color_list[4], "v"],
        ["Greedy + DDPG", "./main_result/learn/GapToSecondPrice/DDPG/", [], color_list[2], "s"],
        ["Greedy + DQN", "./main_result/learn/GapToSecondPrice/DQN/", [], color_list[3], "p"],
    ]
    for name, data_dir, exceptions, color, marker in name_and_paths:
        data_step, mask = draw_avg_curve_for_method(plt, data_dir, name, marker, False, exceptions, color,
                                                    use_move_avg=False,
                                                    show_variance=False)

    other_datas = [

    ]
    show_other_fixed_data(plt, other_datas, data_step, mask)

    plt.xlabel('Training Episode')
    plt.ylabel('Gap to Market Second Price')

    plt.legend(frameon=True, fontsize=23)

    plt.show()


def roi_line_crop():
    import sys
    from io import StringIO

    TESTDATA = StringIO(
        '''
        ds	cost+	roi+	category
        2019-12-06 00:00:00	0.05482974314180877	0.08776609639555488	test2
        2019-12-06 00:00:00	-0.015392290982400914	0.16351849067335378	test1
        2019-12-06 00:00:00	0.0	0.0	base
        2019-12-06 00:00:00	0.05725800108579282	0.1089678220013981	test3
        2019-12-07 00:00:00	-0.0055588047141647	0.04423626507206646	test2
        2019-12-07 00:00:00	0.0	0.0	base
        2019-12-07 00:00:00	-0.018047399751649373	0.046134196665059024	test1
        2019-12-07 00:00:00	-0.0064139638723719195	0.12908210562980016	test3
        2019-12-08 00:00:00	0.03450440429398949	0.06279881866779058	test3
        2019-12-08 00:00:00	0.03985448806904657	0.050846230285296556	test2
        2019-12-08 00:00:00	0.0	0.0	base
        2019-12-08 00:00:00	0.002066377327658575	0.11449069726152672	test1
        2019-12-09 00:00:00	-0.033297555626769615	0.06770559995063063	test1
        2019-12-09 00:00:00	0.0	0.0	base
        2019-12-09 00:00:00	0.005428219554454383	0.02604572495146784	test3
        2019-12-09 00:00:00	0.008813467275854414	0.011903381067994934	test2
        2019-12-10 00:00:00	0.02211156613589771	0.027875528316689824	test2
        2019-12-10 00:00:00	-0.03346709325503139	0.03223190125748743	test1
        2019-12-10 00:00:00	0.0	0.0	base
        2019-12-10 00:00:00	0.0192427639853201	0.024349333530767625	test3
        2019-12-11 00:00:00	0.061431792253170414	0.030693846170713135	test2
        2019-12-11 00:00:00	0.0	0.0	base
        2019-12-11 00:00:00	0.060820891611863104	0.042688631920762665	test3
        2019-12-11 00:00:00	-0.02678032903636629	0.031484118462715927	test1
        2019-12-12 00:00:00	0.09951400543985534	0.020637819501926558	test2
        2019-12-12 00:00:00	-0.015075384188008911	0.03179350375386458	test1
        2019-12-12 00:00:00	0.0	0.0	base
        2019-12-12 00:00:00	0.09980756856805928	0.0019829363241936626	test3
        2019-12-13 00:00:00	0.0	0.0	base
        2019-12-13 00:00:00	0.006197092786760505	0.03584641039875702	test1
        2019-12-13 00:00:00	0.00802383095507242	0.09502867179452412	test2
        2019-12-13 00:00:00	0.008172819847807267	0.06384245776844977	test3
        2019-12-14 00:00:00	0.0	0.0	base
        2019-12-14 00:00:00	-0.0440801008150683	0.02967109431306736	test1
        2019-12-14 00:00:00	0.06279384485148953	0.005756662560236103	test2
        2019-12-14 00:00:00	0.07210908322722354	-0.0015274548502316465	test3
        2019-12-15 00:00:00	-0.0000025860963467350118	0.009144666417654435	test1
        2019-12-15 00:00:00	-0.008270523128326768	0.1664594252059901	test2
        2019-12-15 00:00:00	-0.003941834019015711	0.0718238411965566	test3
        2019-12-15 00:00:00	0.0	0.0	base
        2019-12-16 00:00:00	0.0	0.0	base
        2019-12-16 00:00:00	-0.015071154592149694	0.10260606847425935	test1
        2019-12-16 00:00:00	-0.0002906632799001274	0.07740007930237258	test2
        2019-12-16 00:00:00	-0.03138436818910484	0.1069446135058374	test3
        2019-12-17 00:00:00	0.0	0.0	base
        2019-12-17 00:00:00	0.015489917779937468	0.07468599079488358	test2
        2019-12-17 00:00:00	-0.013636186227054314	0.028504219909729178	test1
        2019-12-17 00:00:00	0.025321172051891327	0.029843021270490988	test3
        2019-12-18 00:00:00	0.0	0.0	base
        2019-12-18 00:00:00	-0.09483432594132302	0.06985073062794878	test1
        2019-12-18 00:00:00	-0.03120500231522172	0.19073351075403244	test2
        2019-12-18 00:00:00	-0.023650755648689126	0.09146187881956691	test3
        2019-12-19 00:00:00	-0.03905023125123075	-0.0016479817800431062	test1
        2019-12-19 00:00:00	-0.025829584036777598	0.09316271294564848	test2
        2019-12-19 00:00:00	-0.03517375967512848	0.10549030932598669	test3
        2019-12-19 00:00:00	0.0	0.0	base
        2019-12-20 00:00:00	0.0	0.0	base
        2019-12-20 00:00:00	-0.0355373265429636	0.0977568096692989	test2
        2019-12-20 00:00:00	-0.032437416253926266	0.08208794479548143	test1
        2019-12-20 00:00:00	-0.028772095723518354	0.017985084142390972	test3
        2019-12-21 00:00:00	-0.011790368073389956	-0.017763190653143912	test1
        2019-12-21 00:00:00	-0.04127892396718813	0.14869371072756388	test2
        2019-12-21 00:00:00	-0.0510908445341397	0.1716249707215376	test3
        2019-12-21 00:00:00	0.0	0.0	base
        2019-12-22 00:00:00	-0.03766234447201422	-0.0677393278548134	test1
        2019-12-22 00:00:00	-0.03664120114751712	0.05352305003259206	test2
        2019-12-22 00:00:00	-0.04360016042027537	0.09848655965732367	test3
        2019-12-22 00:00:00	0.0	0.0	base
        2019-12-23 00:00:00	-0.022283540691462878	0.00557119366291392	test1
        2019-12-23 00:00:00	0.0	0.0	base
        2019-12-23 00:00:00	-0.026775140230451577	0.0974411446658896	test2
        2019-12-23 00:00:00	-0.028045033811351927	0.09174133503210746	test3
        2019-12-24 00:00:00	-0.02595063196773917	0.06498529929574515	test3
        2019-12-24 00:00:00	0.0012884858989703485	0.024739015104974893	test1
        2019-12-24 00:00:00	0.0	0.0	base
        2019-12-24 00:00:00	-0.02633154939908855	0.12440076557509672	test2
        2019-12-25 00:00:00	0.012379547381101474	0.035383046212991376	test3
        2019-12-25 00:00:00	0.0	0.0	base
        2019-12-25 00:00:00	0.007842363318261825	0.09676833088672443	test2
        2019-12-25 00:00:00	-0.05079349781151121	0.017002281683709874	test1
        2019-12-26 00:00:00	-0.03462601829011769	-0.008647827245796824	test1
        2019-12-26 00:00:00	0.0	0.0	base
        2019-12-26 00:00:00	0.0004429622131632005	0.11787974474086771	test3
        2019-12-26 00:00:00	-0.005649586500383941	0.0951509465523579	test2
        '''
    )

    df = pd.read_csv(TESTDATA, sep="\t")
    df.ds = pd.to_datetime(df.ds)

    fig, ax = plt.subplots(figsize=[15, 5])
    plt.grid(linestyle="--")

    part = df.loc[df.ds >= pd.to_datetime('20191210')].loc[df.ds <= pd.to_datetime('20191220')]

    ax.plot(range(int(len(part) / 4)), part.loc[part.category == 'test1']['roi+'], '-o', markersize=12, linewidth=4,
            label='Contextual Bandit')
    ax.plot(range(int(len(part) / 4)), part.loc[part.category == 'test2']['roi+'], '-s', markersize=12, linewidth=4,
            label='MSBCB')
    ax.xaxis.set_ticks(range(int(len(part) / 4)))

    ax.legend(loc=2)

    days = ["Day %d" % i for i in range(int(len(part) / 4))]
    ax.set_xticklabels(days, rotation=0)

    vals = ax.get_yticks()
    ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    ax.set_ylabel('% Improvement in ROI')

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main_sub_fiture_1()
