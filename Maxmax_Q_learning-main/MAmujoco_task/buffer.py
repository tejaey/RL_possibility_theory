import numpy as np
import torch



class ReplayBuffer(object):
    def __init__(self, args):
        self.args = args
        self.buffer_size = args.buffer_size
        self.n_ant = args.n_agent
        self.state_space = args.state_space
        self.action_dim = args.action_dim
        self.pointer = 0
        self.len = 0
        self.states = np.zeros((self.buffer_size, self.state_space))
        self.actions = np.zeros((self.buffer_size, self.action_dim))
        self.rewards = np.zeros((self.buffer_size,1))
        self.next_states = np.zeros((self.buffer_size, self.state_space))
        self.dones = np.zeros((self.buffer_size,1))
#        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(args.device)

    # 增加变成tensor的指令
    def getBatch(self, batch_size):
        if self.len < batch_size*1000:
            index = np.random.choice(self.len, batch_size, replace=False)
        else:
            index = np.random.choice(self.len, batch_size, replace=True)
        return ( torch.FloatTensor(self.states[index]).to(self.device), torch.FloatTensor(self.actions[index]).to(self.device), torch.FloatTensor(self.rewards[index]).to(self.device), torch.FloatTensor(self.next_states[index]).to(self.device), torch.FloatTensor(self.dones[index]).to(self.device) )

    def add(self, state, action, reward, next_state, done):

        self.actions[self.pointer] = action
        self.rewards[self.pointer] = reward
        self.dones[self.pointer] = done
        self.states[self.pointer] = state
        self.next_states[self.pointer] = next_state
        self.dones[self.pointer] = done
        self.pointer = (self.pointer + 1)%self.buffer_size
        self.len = min(self.len + 1, self.buffer_size)
