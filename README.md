# Dynamic Knapsack Optimization Towards Efficient Multi-Channel Sequential Advertising

This is the code implementation for the **(1) simulation environment**, **(2) MSBCB framework** and **(3) all compared baselines** presented in the paper: Dynamic Knapsack Optimization Towards Efficient Multi-Channel Sequential Advertising. 

## 1. Code structure
* **./requirements.txt:** `including the modules/packages on which the program depends. These pakages should be installed before runing the code bellow.`
* **./agents:** `core code for our MSBCB framework and all compared baseline algorithms.`
* **./simulation_env:** `the code for the virtual environment.`
* **./replay_buffer:** `the code of the experience replay buffers for reinforcement learning algorithms.`
* **./plot_util:** `the code for the tensorboard-logger.`
* **./figure_for_paper:** `the code for drawing figures.`


## 2. Run the code
```
cd ./agents

python msbcb.py --seed=1 --user_num=10000 --budget=12000 --init_cpr_thr=6.
python greedy_with_dqn.py --seed=1 --user_num=10000 --budget=12000 --init_cpr_thr=6.
python greedy_with_ddpg.py --seed=1 --user_num=10000 --budget=12000 --init_cpr_thr=6.
python greedy_with_ppo.py --seed=1 --user_num=10000 --budget=12000 --init_cpr_thr=6.
python greedy_with_max_cpr.py --seed=1 --user_num=10000 --budget=12000 --init_cpr_thr=6.
python contextual_bandit.py --seed=1 --user_num=10000 --budget=12000 --init_cpr_thr=6.
python constrained_dqn.py --seed=1 --user_num=10000 --budget=12000 --init_cpr_thr=6.
python constrained_ddpg.py --seed=1 --user_num=10000 --budget=12000 --init_cpr_thr=6.
python constrained_ppo.py --seed=1 --user_num=10000 --budget=12000 --init_cpr_thr=6.
python offline_optimal.py --seed=1 --user_num=10000 --budget=12000 --init_cpr_thr=6.

```

