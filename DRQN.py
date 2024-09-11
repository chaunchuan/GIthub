import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import model
import xlwt
import time
import collections
import openpyxl

reward_shaping  = True
showFigures     = True
printDetails    = True
############################# 设定迭代参数衰减率 #############################
LearnStep = 0


############################# 设定迭代的总轮数 #############################
episodes        = 20

############################# 导入环境文件、是否使用cuda #############################
env = model.model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

############################# 定义一个用于强化学习的神经网络 #############################
class Q_net(nn.Module):
    def __init__(self, state_space=3,
                 action_space=25410):
        super(Q_net, self).__init__()

        assert state_space is not None, "None state_space input: state_space should be selected."
        assert action_space is not None, "None action_space input: action_space should be selected."

        self.hidden_space = 128
        self.state_space = state_space
        self.action_space = action_space

        self.Linear1 = nn.Linear(self.state_space, self.hidden_space)
        self.lstm = nn.LSTM(self.hidden_space, self.hidden_space, batch_first=True)
        self.Linear2 = nn.Linear(self.hidden_space, self.action_space)

    def forward(self, x, h, c):
        x = F.relu(self.Linear1(x))
        x, (new_h, new_c) = self.lstm(x, (h, c))
        x = self.Linear2(x)
        return x, new_h, new_c

    def sample_action(self, obs, h, c,  epsilon ):
        output, new_h, new_c = self.forward(obs, h, c)
        #print(epsilon)

        if random.random() < epsilon:
            return random.randrange(0,25410), new_h, new_c
        else:
            index = output.argmax().item()
            return index, new_h, new_c

    def init_hidden_state(self, batch_size, training=None):

        assert training is not None, "training step parameter should be dtermined"

        if training is True:
            return torch.zeros([1, batch_size, self.hidden_space]), torch.zeros([1, batch_size, self.hidden_space])
        else:
            return torch.zeros([1, 1, self.hidden_space]), torch.zeros([1, 1, self.hidden_space])

#这个 EpisodeBuffer 类是一个简单的numpy重播缓冲区，用于存储智能体的经验回放数据
class EpisodeBuffer:

    def __init__(self):
        self.obs = []
        self.action = []
        self.reward = []
        self.next_obs = []
        self.done = []

    def put(self, transition):
        self.obs.append(transition[0])
        self.action.append(transition[1])
        self.reward.append(transition[2])
        self.next_obs.append(transition[3])
        self.done.append(transition[4])

    def sample(self, random_update=True, lookup_step=None, idx=None):
        obs = np.array(self.obs)
        action = np.array(self.action)
        reward = np.array(self.reward)
        next_obs = np.array(self.next_obs)
        done = np.array(self.done)

        if random_update is True:
            obs = obs[idx:idx+lookup_step]
            action = action[idx:idx+lookup_step]
            reward = reward[idx:idx+lookup_step]
            next_obs = next_obs[idx:idx+lookup_step]
            done = done[idx:idx+lookup_step]

        return dict(obs=obs,
                    acts=action,
                    rews=reward,
                    next_obs=next_obs,
                    done=done)

    def __len__(self) -> int:
        return len(self.obs)

#这段代码定义了一个名为 EpisodeMemory 的类，用于管理递归智能体的经验记忆
#这段代码的主要目的是从经验记忆中采样数据以供训练使用，根据 random_update 参数的不同，采用不同的采样方式。
class EpisodeMemory():
#max_epi_num：整数，表示最大的经验片段数量、max_epi_len：整数，表示每个经验片段的最大长度、batch_size：整数，表示每次采样的批量大小
#lookup_step：整数，表示用于采样的查找步长。 在初始化过程中，将这些参数存储在实例变量中，并创建一个最大长度为 max_epi_num 的双向队列 memory，用于存储经验片段
    def __init__(self, random_update=True,
                 max_epi_num=8000, max_epi_len=240,
                 batch_size=48,
                 lookup_step=None):
        self.random_update = random_update  # if False, sequential update
        self.max_epi_num = max_epi_num
        self.max_epi_len = max_epi_len
        self.batch_size = batch_size
        self.lookup_step = lookup_step

        self.memory = collections.deque(maxlen=self.max_epi_num)

    def put(self, episode):
        self.memory.append(episode)

    def sample(self):
        sampled_buffer = []
#如果 min_step 大于 lookup_step，则使用 lookup_step 进行采样，并将样本添加到 sampled_buffer 中。否则，使用 min_step 进行采样，并将样本添加到 sampled_buffer 中。
#随机更新方式
        if self.random_update:
            sampled_episodes = random.sample(self.memory, self.batch_size)

            check_flag = True
            min_step = self.max_epi_len

            for episode in sampled_episodes:
                min_step = min(min_step, len(episode))

            for episode in sampled_episodes:
                if min_step > self.lookup_step:
                    idx = np.random.randint(0, len(episode) - self.lookup_step + 1)
                    sample = episode.sample(random_update=self.random_update, lookup_step=self.lookup_step, idx=idx)
                    sampled_buffer.append(sample)
                else:
                    idx = np.random.randint(0, len(episode) - min_step + 1)
                    sample = episode.sample(random_update=self.random_update, lookup_step=min_step, idx=idx)
                    sampled_buffer.append(sample)
#顺序更新方式
        else:
            idx = np.random.randint(0, len(self.memory))
            sampled_buffer.append(self.memory[idx].sample(random_update=self.random_update))

        return sampled_buffer, len(sampled_buffer[0]['obs'])

    def __len__(self):
        return len(self.memory)


#这段代码的作用是从经验回放缓冲区中抽取样本，计算 Q-learning 的目标值，然后使用当前 Q 网络计算当前状态的 Q 值，并且根据实际执行的动作选择相应的 Q 值
def train(q_net=None, target_q_net=None, episode_memory=None,
          device=None,
          optimizer=None,
          batch_size=48,
          learning_rate=1e-3,
          gamma=0.01):
    assert device is not None, "None Device input: device should be selected."


#从经验回放缓冲区（replay buffer）中抽取一个批次的样本。episode_memory 是一个经验回放缓冲区对象，sample() 方法用于随机抽取样本，seq_len 是每个样本序列的长度。
    samples, seq_len = episode_memory.sample()
#代码将样本中的观察（observations）、动作（actions）、奖励（rewards）、下一个观察（next_observations）和完成标志（dones）分别存储到对应的列表中。
    observations = []
    actions = []
    rewards = []
    next_observations = []
    dones = []

    for i in range(batch_size):
        observations.append(samples[i]["obs"])
        actions.append(samples[i]["acts"])
        rewards.append(samples[i]["rews"])
        next_observations.append(samples[i]["next_obs"])
        dones.append(samples[i]["done"])
#将这些列表转换为 NumPy 数组，并通过 PyTorch 的 torch.FloatTensor() 和 torch.LongTensor() 方法将它们转换为张量（tensors）。这些张量将被送入神经网络进行训练
    observations = np.array(observations)
    actions = np.array(actions)
    rewards = np.array(rewards)
    next_observations = np.array(next_observations)
    dones = np.array(dones)

    observations = torch.FloatTensor(observations.reshape(batch_size, seq_len, -1)).to(device)
    actions = torch.LongTensor(actions.reshape(batch_size, seq_len, -1)).to(device)
    rewards = torch.FloatTensor(rewards.reshape(batch_size, seq_len, -1)).to(device)
    next_observations = torch.FloatTensor(next_observations.reshape(batch_size, seq_len, -1)).to(device)
    dones = torch.FloatTensor(dones.reshape(batch_size, seq_len, -1)).to(device)
#初始化目标 Q 网络的隐藏状态
    h_target, c_target = target_q_net.init_hidden_state(batch_size=batch_size, training=True)
#使用目标 Q 网络计算下一个状态的 Q 值 q_target，并选择其中的最大值 q_target_max。这里的目标 Q 网络是用于计算目标 Q 值的网络
    q_target, _, _ = target_q_net(next_observations, h_target.to(device), c_target.to(device))
    q_target_max = q_target.max(2)[0].view(batch_size, seq_len, -1).detach()
#计算目标 Q 值 targets，它是当前奖励加上未来预期奖励的折扣值
    targets = rewards + gamma * q_target_max * dones
#使用当前 Q 网络计算当前状态的 Q 值 q_out，并根据实际执行的动作 actions 从中选择相应的 Q 值 q_a
    h, c = q_net.init_hidden_state(batch_size=batch_size, training=True)
    q_out, _, _ = q_net(observations, h.to(device), c.to(device))
    q_a = q_out.gather(2, actions)
#这段代码的作用是计算 Q-learning 的损失函数，然后通过反向传播和优化器来更新模型的参数，使得模型的预测值能够逼近目标值，从而提高模型的性能
    loss = F.smooth_l1_loss(q_a, targets)
    #loss = nn.MSELoss(q_a, targets)
#Update Network
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())




if __name__ == "__main__":
#action = np.load("action_all.npy", allow_pickle=True) 这里加载了一个名为 "action_all.npy" 的 NumPy 数组文件，将其存储在变量 action 中
    action = np.load("action_all.npy", allow_pickle=True)
    random_update = True
    #max_epi_len = 300


#创建了一个 Q 网络，这是强化学习中用于估计动作值函数的神经网络模型
    Q = Q_net(state_space=3,
              action_space=25410).to(device)
#创建了一个目标 Q 网络，用于计算目标 Q 值
    Q_target = Q_net(state_space=3,
                     action_space=25410).to(device)
# 将 Q 网络的权重复制给目标 Q 网络，初始化它们的权重相同
    Q_target.load_state_dict(Q.state_dict())
# 设定优化器中的系数
    batch_size = 48
    learning_rate=0.001
    score = 0
    score_sum = 0
# 创建优化器，更新Q网络的权重
    optimizer = optim.Adam(Q.parameters(), lr=learning_rate)
# 创建了一个用于存储经验回放（experience replay）的对象。这是为了在训练中使用记忆回放，提高训练效率和稳定性。
    episode_memory = EpisodeMemory(random_update=random_update,
                                   max_epi_num=8000, max_epi_len=240,
                                   batch_size=batch_size,
                                   lookup_step=36)
# min_epi_num=20 指定了训练开始前要收集的最小 episode 数量
    min_epi_num=8000


    epsilon = 1.5
    eps_end = 0.001
    eps_decay = 0.99992
# Train接下来的部分将是用于训练模型的代码
    for episode in range(episodes):
        obs = env.reset()
        done = False
        print(obs)
#这些变量用于追踪每个回合的奖励、步数等信息
        r = 0
        step = 0
        row = 0
        row_loss = 0


#经验回访、lstm初始话参数更新、衰减率控制动作选择
        episode_record = EpisodeBuffer()
        h, c = Q.init_hidden_state(batch_size=batch_size, training=False)
        t=0

        while True:
            episode_actions = []
            losses = []
            CLs = obs[0]
            a, h, c = Q.sample_action(torch.from_numpy(obs).float().to(device).unsqueeze(0).unsqueeze(0),h.to(device), c.to(device),epsilon)
            T_chws = action[a][0]
            f_tower = action[a][1]
            f_cw_pump = action[a][2]
            f_chw_pump = action[a][3]
            #print(T_chws,f_tower)
            S_, CLc, done, P_chiller, P_tower, R, R_p, R_c, T_chwr, T_outdoor, P_total, Chiller_number,T_cwr = env.step(action[a])

            action_index = int(a.cpu().detach().numpy()) if torch.is_tensor(a) else a
            R = float(R.cpu().detach().numpy()) if torch.is_tensor(R) else R
#奖励值积累设定
            s_prime=S_
            r += R
            obs_prime = s_prime
            # make data
            done_mask = 0.0 if done else 1.0
            episode_record.put([obs, a, R , obs_prime, done_mask])
            #book.save("T_chws - P_chiller " + str(episode) + '.xls')
            s = S_
            obs = obs_prime
            score += r
            score_sum += r

            episode_memory.put(episode_record)
            epsilon = max(eps_end, epsilon * eps_decay)

#首先检查回放缓冲区 episode_memory 的大小是否已达到指定的最小回合数 min_epi_num
            if len(episode_memory) >= min_epi_num:
#如果回放缓冲区的大小满足要求，就调用 train 函数来训练 Q 网络模型 Q 和目标 Q 网络模型 Q_target
                train(Q, Q_target, episode_memory, device,
                      optimizer=optimizer,
                      batch_size=batch_size,
                      learning_rate=learning_rate)

                target_update_period=100
                tau=0.01
#在每个时间步 (t+1)，检查是否需要更新目标 Q 网络 Q_target。如果 (t+1) 是目标更新周期 target_update_period 的倍数，就执行目标网络参数的软更新操作
                if (t + 1) % target_update_period == 0:
                    # Q_target.load_state_dict(Q.state_dict()) <- navie update
                    for target_param, local_param in zip(Q_target.parameters(), Q.parameters()):  # <- soft update
                        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
                t+=1
#如果当前回合已结束（done 为真），则打印出当前回合的编号 episode、回合奖励值 r 以及某种被称为 EPS_Ratio 的比率，并终止当前回合的训练循环
            if done:
                print("Epi:", episode, "Reward:", r, "EPS_Ratio", epsilon)
                break




            episode_actions.append((T_chws, f_tower,f_cw_pump,f_chw_pump))
            file_path = "DRQN-1.txt"

            with open(file_path, "a") as file:
                for step, (T_chws, f_tower,f_cw_pump,f_chw_pump) in enumerate(episode_actions):
                    file.write(f" {P_total:.3f}, {P_chiller:.3f},{P_tower:.3f}, {CLs}, {T_outdoor}, "
                               f"{Chiller_number}, {CLc}, {f_chw_pump}, {T_chws}, {f_cw_pump},{f_tower},{T_cwr:.3f},{T_chwr:.3f},"
                               f"{R_c:.2f},{R_p:.2f}\n")
                file.write("\n")


        # 保存模型的状态字典
        torch.save(Q.state_dict(), 'q_net_model.pth')

