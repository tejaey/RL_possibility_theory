import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario

# two targets with diffferent reward and penalty design. ******* denote the modification 

class Scenario(BaseScenario):
    def make_world(self, num_agents):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = num_agents #3
        self.n_agents = num_agents
        # **********************
        num_landmarks = 2 
        self.num_landmarks = num_landmarks
        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.15
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states # Agents could be anywhere 
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        # ************************* # two targets are in two part: first target in [-1, 0] ; the second target in [0,1]
        for i, landmark in enumerate(world.landmarks):
            ll = [-0.8, 0.4]
            uu = [-0.4, +0.8 ]
            landmark.state.p_pos = np.random.uniform(ll[i], uu[i], world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False


    def reward(self, agent, world):
        rew = 0
        # ******************* 
        # (1) if one agent is in the circle of the first target, no penalty; 
        # (2) two agent are in the circle of the first target, get a small reward
        # (3) if one agnet is in the circle of the second taget, get a smally penalty
        # (4) two agent are in the circle of the second target, get a larger reward
        # *******************
        two_dists = []
        for l in world.landmarks: # two_dists: [ [d1,d2], [d1,d2]]
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            two_dists.append(dists)
        tol = 0.2
        two_arrive = []
        for k in range(self.num_landmarks):
            l = world.landmarks[k] # two_arrive: [ [y1,y2], [y1,y2] ]
            whe_arrive = [dd < (agent.size + l.size +  tol ) for dd in two_dists[k]]
            two_arrive.append(whe_arrive)

        if np.sum(two_arrive[0]) == self.n_agents: # two agents arriving at target 1
            rew -= 2.5
        elif np.sum(two_arrive[1]) == self.n_agents: # two agent arrive at target 2
            rew -= 0
        elif np.sum(two_arrive[1]) == 1: # one agent arrive at target 2
            rew -= 3.5
        else: # two agents are outside two targets or one agent is inside the target 1
            rew -= 3

        # if np.sum(whe_arrive) == self.n_agents:
        #     rew -= 3 * min(dists)
        # elif np.sum(whe_arrive) > 0 :
        #     rew -=  ( 3 * ( agent.size + l.size + tol ) + 0.2)
        # else:
        #     rew -= 3 * ( agent.size + l.size + tol )

        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
#            entity_pos.append(entity.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
#            other_pos.append(other.state.p_pos)
                # [agent.state.p_vel] +
        return np.concatenate( [agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos )
