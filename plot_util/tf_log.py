import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import rcParams
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

params = {'font.family': 'serif',
          'font.serif': 'Times New Roman',
          'font.style': 'normal',  
          'font.weight': 'normal',  
          'font.size': 26,  
          }










rcParams.update(params)
import seaborn as sns


sns.set_style('whitegrid')
sns.set_context(font_scale=3)


class Logger():
    def __init__(self, log_path, sess, exist_writer, build_new_sess):
        
        
        if exist_writer is None:
            self.log_dir = log_path
            if not os.path.exists(os.path.dirname(log_path)):
                os.makedirs(os.path.dirname(log_path))
            self.writer = tf.summary.FileWriter(log_path)
        else:
            self.writer = exist_writer

        
        if build_new_sess:
            gpu_options = tf.GPUOptions(allow_growth=True)
            config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
            sess = tf.Session(config=config, graph=tf.get_default_graph())
        self.sess = sess if sess is not None else tf.get_default_session()
        assert self.sess is not None


class LogScalar(Logger):
    def __init__(self, sess, kws, log_path, exist_writer=None, build_new_sess=False):
        super().__init__(log_path, sess, exist_writer, build_new_sess)
        
        
        self.kws = kws
        self.options = {}
        self.phs = {}
        for key in kws:
            ph = tf.placeholder(dtype=tf.float64, shape=[], name=key + "_ph")
            self.phs[key] = ph
            self.options[key] = tf.summary.scalar(name=key, tensor=ph)

    def log(self, kws, values, global_step, except_kws=None):
        
        for key, value in zip(kws, values):
            
            
            if except_kws is None or (
                    except_kws is not None and isinstance(except_kws, list) and not (key in except_kws)):
                self.writer.add_summary(self.sess.run(
                    self.options[key], {
                        self.phs[key]: value
                    }
                ), global_step=global_step)
        self.writer.flush()

    @staticmethod
    def logCSV(log_dir, seed, csv_path, tag_name_lst=None):
        data = EventAccumulator(log_dir)
        data.Reload()
        
        for tag in data.Tags().get('scalars'):
            print(tag)
            if tag_name_lst is not None and len(tag_name_lst) > 0:
                if tag in tag_name_lst:
                    w_times, step_nums, vals = zip(*data.Scalars(tag))
                    in_data = {'w_times': w_times,
                               'step': step_nums,
                               'value': vals}
                    in_data = pd.DataFrame(in_data)
                    in_data.to_csv("{}/{}_seed_{}.csv".format(csv_path, str(tag), str(seed)), index=False)
            else:
                w_times, step_nums, vals = zip(*data.Scalars(tag))
                in_data = {'wall_time': w_times,
                           'step': step_nums,
                           'value': vals}
                in_data = pd.DataFrame(in_data)
                in_data.to_csv("{}/{}_seed_{}.csv".format(csv_path, str(tag), str(seed)), index=False)

    @staticmethod
    def logTagCSV(log_parent_dir, csv_path, tag_name_lst=None, seed_lst=(1, 2, 3, 4, 5)):
        for seed in seed_lst:
            LogScalar.logCSV("{}/seed_{}".format(log_parent_dir, str(seed)), seed, csv_path, tag_name_lst)

    @staticmethod
    def draw_learning_curves(csv_path, std_coef=1., step=5, x_label='x', y_label='y', alogrithm='alogrithm_name',
                             save_figure_name=None, is_win_rate=False):
        sub_dirs = os.listdir(csv_path)
        data_frames = []
        seed_data_left = pd.read_csv(os.path.join(csv_path, sub_dirs[0]))[["step", "value"]]
        seed_data_left.rename(columns={'value': 'value_0'}, inplace=True)
        for idx, sub_file in enumerate(sub_dirs[1:]):
            seed_data = pd.read_csv(os.path.join(csv_path, sub_file))[["step", "value"]]
            seed_data_left = pd.merge(seed_data_left, seed_data, how="inner", on='step')
            seed_data_left.rename(columns={'value': 'value_{}'.format(idx + 1)}, inplace=True)

        
        data = seed_data_left
        idxs, mean_values, lower_bound, upper_bound = [], [], [], []
        xtick_pos, unit = [], 1000
        for idx in range(0, len(data), step):
            
            avg_idx = int(np.mean(data['step'][idx:idx + step]))
            range_data = data.iloc[idx:idx + step, range(1, data.shape[1])]
            range_data = np.array(range_data)
            mean = np.mean(range_data)
            std = np.std(range_data)
            idxs.append(avg_idx)
            if idx % unit == 0:
                xtick_pos.append(idx)
            mean_values.append(mean)
            
            
            lower_bound.append(np.maximum(mean - std * std_coef, np.min(range_data)))
            upper_bound.append(np.minimum(mean + std * std_coef, np.max(range_data)))
        xtick_pos.append(xtick_pos[-1] + unit)

        
        plt.tight_layout()
        
        xtick_labels = ["{}k".format(int(pos / unit)) if pos > 0 else 0 for pos in xtick_pos]
        
        
        
        plt.xticks()
        if is_win_rate:
            plt.yticks(range(0, 101, 10))
            plt.ylim(min(lower_bound), 100)
        
        

        plt.xlabel(x_label, labelpad=2.5, fontweight=params['font.weight'])
        plt.ylabel(y_label, labelpad=-2.5, fontweight=params['font.weight'])

        plt.plot(idxs, mean_values, label=alogrithm, linewidth=4)
        plt.fill_between(idxs, lower_bound, upper_bound, alpha=0.35)
        plt.legend(loc='lower right', shadow=True)
        

        if save_figure_name is not None:
            plt.savefig("{}.pdf".format(save_figure_name))
        plt.show()

    @staticmethod
    def draw_sub_figure(csv_path, std_coef=1., step=5, x_label='x', y_label='y', alogrithm='alogrithm_name', reset=1):
        sub_dirs = os.listdir(csv_path)
        data_frames = []
        seed_data_left = pd.read_csv(os.path.join(csv_path, sub_dirs[0]))[["step", "value"]]
        seed_data_left.rename(columns={'value': 'value_0'}, inplace=True)
        for idx, sub_file in enumerate(sub_dirs[1:]):
            seed_data = pd.read_csv(os.path.join(csv_path, sub_file))[["step", "value"]]
            seed_data_left = pd.merge(seed_data_left, seed_data, how="inner", on='step')
            seed_data_left.rename(columns={'value': 'value_{}'.format(idx + 1)}, inplace=True)

        
        data = seed_data_left
        
        
        
        
        resize = 1

        idxs, mean_values, lower_bound, upper_bound = [], [], [], []
        
        
        if alogrithm.find("off-policy") != -1:
            step = 250
        for idx in range(0, len(data), step):
            
            avg_idx = int(np.mean(data['step'][idx:idx + step]))
            range_data = data.iloc[idx:idx + step, range(1, data.shape[1])]
            range_data = np.array(range_data)
            mean = np.mean(range_data) * reset
            std = np.std(range_data)
            
            if alogrithm.find("on-policy") != -1:
                avg_idx *= 12

            avg_idx *= resize
            idxs.append(avg_idx)
            
            
            mean_values.append(mean)
            
            
            
            lower_bound.append(np.maximum(mean - std * std_coef, np.min(range_data)))
            upper_bound.append(np.minimum(mean + std * std_coef, np.max(range_data)))

            
            if avg_idx > 100000:
                break
        

        plt.plot(idxs, mean_values, label=alogrithm, linewidth=4)
        plt.fill_between(idxs, lower_bound, upper_bound, alpha=0.35)
        return min(lower_bound)

    @staticmethod
    def draw_multi_curves(csv_paths, algorithm_names, save_figure_name, std_coef=1., step=5, x_label='x', y_label='y',
                          is_win_rate=False, resets=None, std_coefs=None, legend_pos="lower right"):
        if resets is None:
            resets = [1] * len(csv_paths)
            std_coefs = [std_coef] * len(csv_paths)
        lower_bound = 0
        for csv_path, algo_name, reset, std_coef in zip(csv_paths, algorithm_names, resets, std_coefs):
            lower_bound_rtn = LogScalar.draw_sub_figure(csv_path, std_coef=std_coef, step=step, x_label=x_label,
                                                        y_label=y_label, alogrithm=algo_name, reset=reset)
            lower_bound = min(lower_bound_rtn, lower_bound)
            print(algo_name)

        plt.xticks()
        if is_win_rate:
            plt.yticks(range(0, 101, 10))
            plt.ylim(lower_bound, 100)

        plt.xlabel(x_label, labelpad=2.5, fontweight=params['font.weight'])
        plt.ylabel(y_label, labelpad=0, fontweight=params['font.weight'])
        

        plt.legend(loc=legend_pos, shadow=True)
        
        

        

        if save_figure_name is not None:
            plt.savefig("{}.pdf".format(save_figure_name))
        plt.show()


class LogHistogram(Logger):
    def __init__(self, sess, kws, shapes, log_path, exist_writer=None, build_new_sess=False):
        super().__init__(log_path, sess, exist_writer, build_new_sess)
        print("Initial LogHistogram...")
        self.kws = kws
        self.shapes = shapes
        self.options = {}
        self.phs = {}
        for key, shape in zip(kws, shapes):
            ph = tf.placeholder(dtype=tf.float64, shape=[shape], name=key + "_ph")
            self.phs[key] = ph
            self.options[key] = tf.summary.histogram(name=key, values=ph)

    def log(self, kws, values, global_step, except_kws=None):
        for key, value in zip(kws, values):
            
            
            if except_kws is None or (
                    except_kws is not None and isinstance(except_kws, list) and not (key in except_kws)):
                self.writer.add_summary(self.sess.run(
                    self.options[key], {
                        self.phs[key]: value
                    }
                ), global_step=global_step)
        self.writer.flush()

    @staticmethod
    def logCSV(log_dir, seed, csv_path, tag_name_lst=None):
        data = EventAccumulator(log_dir)
        data.Reload()
        
        for tag in data.Tags().get('scalars'):
            print(tag)
            if tag_name_lst is not None and len(tag_name_lst) > 0:
                if tag in tag_name_lst:
                    w_times, step_nums, vals = zip(*data.Scalars(tag))
                    in_data = {'w_times': w_times,
                               'step': step_nums,
                               'value': vals}
                    in_data = pd.DataFrame(in_data)
                    in_data.to_csv("{}/{}_seed_{}.csv".format(csv_path, str(tag), str(seed)), index=False)
            else:
                w_times, step_nums, vals = zip(*data.Scalars(tag))
                in_data = {'wall_time': w_times,
                           'step': step_nums,
                           'value': vals}
                in_data = pd.DataFrame(in_data)
                in_data.to_csv("{}/{}_seed_{}.csv".format(csv_path, str(tag), str(seed)), index=False)

    @staticmethod
    def logTagCSV(log_parent_dir, csv_path, tag_name_lst=None, seed_lst=(1, 2, 3, 4, 5)):
        for seed in seed_lst:
            LogScalar.logCSV("{}/seed_{}".format(log_parent_dir, str(seed)), seed, csv_path, tag_name_lst)

    @staticmethod
    def draw_learning_curves(csv_path, std_coef=1., step=5, x_label='x', y_label='y', alogrithm='algorithm_name',
                             save_figure_name=None, is_win_rate=False):
        sub_dirs = os.listdir(csv_path)
        data_frames = []
        seed_data_left = pd.read_csv(os.path.join(csv_path, sub_dirs[0]))[["step", "value"]]
        seed_data_left.rename(columns={'value': 'value_0'}, inplace=True)
        for idx, sub_file in enumerate(sub_dirs[1:]):
            seed_data = pd.read_csv(os.path.join(csv_path, sub_file))[["step", "value"]]
            seed_data_left = pd.merge(seed_data_left, seed_data, how="inner", on='step')
            seed_data_left.rename(columns={'value': 'value_{}'.format(idx + 1)}, inplace=True)

        
        data = seed_data_left
        idxs, mean_values, lower_bound, upper_bound = [], [], [], []
        xtick_pos, unit = [], 1000
        for idx in range(0, len(data), step):
            
            avg_idx = int(np.mean(data['step'][idx:idx + step]))
            range_data = data.iloc[idx:idx + step, range(1, data.shape[1])]
            range_data = np.array(range_data)
            mean = np.mean(range_data)
            std = np.std(range_data)
            idxs.append(avg_idx)
            if idx % unit == 0:
                xtick_pos.append(idx)
            mean_values.append(mean)
            
            
            lower_bound.append(np.maximum(mean - std * std_coef, np.min(range_data)))
            upper_bound.append(np.minimum(mean + std * std_coef, np.max(range_data)))
        xtick_pos.append(xtick_pos[-1] + unit)

        
        plt.tight_layout()
        
        xtick_labels = ["{}k".format(int(pos / unit)) if pos > 0 else 0 for pos in xtick_pos]
        
        
        
        plt.xticks()
        if is_win_rate:
            plt.yticks(range(0, 101, 10))
            plt.ylim(min(lower_bound), 100)
        
        

        plt.xlabel(x_label, labelpad=2.5, fontweight=params['font.weight'])
        plt.ylabel(y_label, labelpad=-2.5, fontweight=params['font.weight'])

        plt.plot(idxs, mean_values, label=alogrithm, linewidth=4)
        plt.fill_between(idxs, lower_bound, upper_bound, alpha=0.35)
        plt.legend(loc='lower right', shadow=True)
        

        if save_figure_name is not None:
            plt.savefig("{}.pdf".format(save_figure_name))
        plt.show()

    @staticmethod
    def draw_sub_figure(csv_path, std_coef=1., step=5, x_label='x', y_label='y', alogrithm='alogrithm_name', reset=1):
        sub_dirs = os.listdir(csv_path)
        data_frames = []
        seed_data_left = pd.read_csv(os.path.join(csv_path, sub_dirs[0]))[["step", "value"]]
        seed_data_left.rename(columns={'value': 'value_0'}, inplace=True)
        for idx, sub_file in enumerate(sub_dirs[1:]):
            seed_data = pd.read_csv(os.path.join(csv_path, sub_file))[["step", "value"]]
            seed_data_left = pd.merge(seed_data_left, seed_data, how="inner", on='step')
            seed_data_left.rename(columns={'value': 'value_{}'.format(idx + 1)}, inplace=True)

        data = seed_data_left


        resize = 1

        idxs, mean_values, lower_bound, upper_bound = [], [], [], []

        
        if alogrithm.find("off-policy") != -1:
            step = 250
        for idx in range(0, len(data), step):
            
            avg_idx = int(np.mean(data['step'][idx:idx + step]))
            range_data = data.iloc[idx:idx + step, range(1, data.shape[1])]
            range_data = np.array(range_data)
            mean = np.mean(range_data) * reset
            std = np.std(range_data)
            
            if alogrithm.find("on-policy") != -1:
                avg_idx *= 12

            avg_idx *= resize
            idxs.append(avg_idx)

            mean_values.append(mean)

            lower_bound.append(np.maximum(mean - std * std_coef, np.min(range_data)))
            upper_bound.append(np.minimum(mean + std * std_coef, np.max(range_data)))

            
            if avg_idx > 100000:
                break


        plt.plot(idxs, mean_values, label=alogrithm, linewidth=4)
        plt.fill_between(idxs, lower_bound, upper_bound, alpha=0.35)
        return min(lower_bound)

    @staticmethod
    def draw_multi_curves(csv_paths, algorithm_names, save_figure_name, std_coef=1., step=5, x_label='x', y_label='y',
                          is_win_rate=False, resets=None, std_coefs=None, legend_pos="lower right"):
        if resets is None:
            resets = [1] * len(csv_paths)
            std_coefs = [std_coef] * len(csv_paths)
        lower_bound = 0
        for csv_path, algo_name, reset, std_coef in zip(csv_paths, algorithm_names, resets, std_coefs):
            lower_bound_rtn = LogScalar.draw_sub_figure(csv_path, std_coef=std_coef, step=step, x_label=x_label,
                                                        y_label=y_label, alogrithm=algo_name, reset=reset)
            lower_bound = min(lower_bound_rtn, lower_bound)
            print(algo_name)

        plt.xticks()
        if is_win_rate:
            plt.yticks(range(0, 101, 10))
            plt.ylim(lower_bound, 100)

        plt.xlabel(x_label, labelpad=2.5, fontweight=params['font.weight'])
        plt.ylabel(y_label, labelpad=0, fontweight=params['font.weight'])


        plt.legend(loc=legend_pos, shadow=True)


        if save_figure_name is not None:
            plt.savefig("{}.pdf".format(save_figure_name))
        plt.show()


def reload_data(path):
    with open(path, 'rb') as fp:
        data = pickle.load(fp)
        return data


if __name__ == "__main__":
    pass
