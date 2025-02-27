import numpy as np
from make_env import make_env


class Sequential_Target_Env:
    def __init__(self, args):
        scenario = "simple_spread_sequential"
        self.env = make_env(scenario, args.n_agent)
        self.args = args
        self.n_agent = args.n_agent
        self.state_space = 11
        self.action_dim = self.env.action_space[0].n

        self.reward_TwoInT = args.reward_TwoInT

    def step(self, action):
        # obs : [agent1.state.p_vel, agent2.state.p_vel, agent1.state.p_pos, agent2.state.p_pos, entity1_pos, entity2_pos ]
        next_obs, _ ,  terminated, info  = self.env.step(action)
        n_obs = next_obs[0]

        # calculate the reward and extra dimension (item_weight)
        agent1_vel = np.array([n_obs[0], n_obs[1]])
        agent2_vel = np.array([n_obs[2], n_obs[3]])
        agent1_pos = np.array([n_obs[4], n_obs[5]])
        agent2_pos = np.array([n_obs[6], n_obs[7]])
        target1_pos = np.array([n_obs[8], n_obs[9]])
        target2_pos = np.array([n_obs[10], n_obs[11]])

        reward = 0
        two_dists = []
        two_arrive = []
        c1_long = 0.4
        c2_long = 0.4

        # two_dists[0]  distances between two agents and the first target
        two_dists.append([ np.sqrt(np.sum(np.square(agent1_pos - target1_pos))), np.sqrt(np.sum(np.square(agent2_pos - target1_pos)))])
        two_dists.append( [np.sqrt(np.sum(np.square(agent1_pos - target2_pos))), np.sqrt(np.sum(np.square(agent2_pos - target2_pos)))])

        whe_arrive1 = [dd < c1_long for dd in two_dists[0]]
        whe_arrive2= [dd < c2_long for dd in two_dists[1]]
        two_arrive.append(whe_arrive1)
        two_arrive.append(whe_arrive2)

        

        if np.sum(two_arrive[0]) == self.n_agent:
            reward -= 6
            self.item_w = 0.5

        elif np.sum(two_arrive[1]) == self.n_agent:
            reward -= 3
            reward += self.item_w * 5


        elif ( np.sum(two_arrive[1]) > 0 )  and ( np.sum(two_arrive[1]) < self.n_agent ): # only one agent in target 2
            reward -= 6.5
        else:
            reward -= 6

        rew = [reward, reward]
        next_obs1 = np.concatenate([agent1_vel] + [agent1_pos] + [agent1_pos - target1_pos]  +  [agent1_pos - target2_pos] + [agent1_pos - agent2_pos] + [np.array([self.item_w])] )
        next_obs2 = np.concatenate([agent2_vel] + [agent2_pos] + [agent2_pos - target1_pos]  +  [agent2_pos - target2_pos] + [agent2_pos - agent1_pos] + [np.array([self.item_w])] )
        

        next_obs = [next_obs1, next_obs2]

        return next_obs, rew, terminated, info
    
    def reset(self):
        self.item_w = 0.0

        nnn_obs = self.env.reset()
        n_obs = nnn_obs[0]

        agent1_vel = np.array([n_obs[0], n_obs[1]])
        agent2_vel = np.array([n_obs[2], n_obs[3]])
        agent1_pos = np.array([n_obs[4], n_obs[5]])
        agent2_pos = np.array([n_obs[6], n_obs[7]])
        target1_pos = np.array([n_obs[8], n_obs[9]])
        target2_pos = np.array([n_obs[10], n_obs[11]])


        obs1 = np.concatenate([agent1_vel] + [agent1_pos] + [agent1_pos - target1_pos]  +  [agent1_pos - target2_pos] + [agent1_pos - agent2_pos] + [ np.array([self.item_w])] )
        obs2 = np.concatenate([agent2_vel] + [agent2_pos] + [agent2_pos - target1_pos]  +  [agent2_pos - target2_pos] + [agent2_pos - agent1_pos] + [ np.array([self.item_w])] )
        
        obs = [obs1, obs2]

        return obs

        
