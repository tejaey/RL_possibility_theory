import numpy as np
import copy

def f(x, m, sub_reward):
    if x < m:
        return 0.5 * (np.cos(x * np.pi / m ) + 1)
    elif x < 0.6:
        return 0
    elif x < 1.0:
        return sub_reward * (np.cos( 5 * np.pi * ( x - 0.8 )) + 1  )
    else:
        return 0

def r_f(x, m, sub_reward):
    tx = copy.deepcopy(x)
    x[tx<m] = 0.5 * (np.cos(x[tx<m] * np.pi / m ) + 1)
    x[(tx<=0.6) & (tx > m)] = 0
    x[(tx<1.0) & (tx > 0.6)] = sub_reward * (np.cos( 5 * np.pi * ( x[(tx <1.0) & (tx > 0.6)] - 0.8 )) + 1  )
    x[tx>=1.0] = 0
    return x


class World(object):
    def __init__(self, args):
        super(World, self).__init__()
        self.args = args
        self.n_agent = args.n_agent
        self.n_action = args.action_dim
        self.len_state = args.state_space
        self.m = args.m
        self.sub_reward = args.sub_reward
#        self.m_list = [0.0,0.1,0.12,0.25,0.38,0.51]
#        self.m = self.m_list[n]
        if self.args.set_env_seed:
            np.random.seed(args.seed)
        self.x = np.random.rand(self.n_agent)*2-1

    def reset(self):
        self.x = np.random.rand(self.n_agent)*2-1

        return self.x

    def get_state(self):

        return self.x


    def step(self, actions):
        done = False
        self.x = np.clip(self.x + 0.1*actions,-1,1)
        reward = f(np.sqrt(2/self.n_agent*sum(self.x**2)), self.m, self.sub_reward)

        return self.x, reward, done
