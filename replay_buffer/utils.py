import pickle


def discount_with_dones(trajectory, gamma):
    '''
    :param trajectory: (obs_t, action, reward, obs_tp1, done)
    :param gamma:
    :return:
    '''
    discounted = []
    r = 0
    
    normal_return = 0
    for item in trajectory[::-1]:
        r = item[2] + gamma * r * (1. - item[-1])  
        normal_return = item[2] + normal_return * (1. - item[-1])
        discounted.append(r)
    return discounted[::-1], normal_return


def add_episode(buffer, trajectory, gamma):
    returns, Return = discount_with_dones(trajectory, gamma)
    
    end = len(returns) - 1
    start = 0
    for (obs_t, action, reward, obs_tp1, done), R in zip(trajectory, returns):
        buffer.add(obs_t, action, reward, obs_tp1, float(done), float(end - start), R)
        start += 1
    buffer.mean_returns.append(buffer.current_mean_return)


def add_positive_episode(buffer, trajectory, gamma, mean=0):
    returns, Return = discount_with_dones(trajectory, gamma)
    if returns[-1] >= mean:
        end = len(returns) - 1
        start = 0
        for (obs_t, action, reward, obs_tp1, done), R in zip(trajectory, returns):
            buffer.add(obs_t, action, reward, obs_tp1, float(done), float(end - start), R)
            start += 1
        buffer.mean_returns.append(buffer.current_mean_return)


def add_trajectory(buffer, trajectory, gamma):
    returns, Return = discount_with_dones(trajectory, gamma)
    end = len(returns) - 1
    start = 0
    new_trajectory = []
    for (obs_t, action, reward, obs_tp1, done), R in zip(trajectory, returns):
        new_trajectory.append([obs_t, action, reward, obs_tp1, float(done), float(end - start), R])
        start += 1
    buffer.add(new_trajectory, Return)

def discount_with_dones_with_p(trajectory, gamma):
    '''
    :param trajectory: (obs_t, action, p, reward, obs_tp1, done)
    :param gamma:
    :return:
    '''
    discounted = []
    r = 0
    
    normal_return = 0
    for item in trajectory[::-1]:
        obs_t, action, p, reward, obs_tp1, done = item
        r = reward + gamma * r * (1. - item[-1])
        normal_return = item[2] + normal_return * (1. - item[-1])
        discounted.append(r)
    return discounted[::-1], normal_return


def add_trajectory_with_p(buffer, trajectory, gamma):
    returns, Return = discount_with_dones_with_p(trajectory, gamma)
    end = len(returns) - 1
    start = 0
    new_trajectory = []
    for (obs_t, action, reward, obs_tp1, done), R in zip(trajectory, returns):
        new_trajectory.append([obs_t, action, reward, obs_tp1, float(done), float(end - start), R])
        start += 1
    buffer.add(new_trajectory, Return)




def add_positive_trajectory(buffer, trajectory, gamma, mean=0):
    returns, Return = discount_with_dones(trajectory, gamma)

    end = len(returns) - 1
    start = 0
    new_trajectory = []
    for (obs_t, action, reward, obs_tp1, done), R in zip(trajectory, returns):
        new_trajectory.append([obs_t, action, reward, obs_tp1, float(done), float(end - start), R])
        start += 1
    if returns[-1] >= mean:
        
        

        buffer.add(new_trajectory, returns[-1])



def reload_data(path):
    with open(path, 'rb') as fp:
        data = pickle.load(fp)
        return data
