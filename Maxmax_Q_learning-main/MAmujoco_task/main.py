import os, sys, time
import numpy as np
import copy
import torch
import wandb
from I2Q_model import I2Q_Agent
from MMQ_model_Quantile import MMQ_Q_Agent
from buffer import ReplayBuffer
import argparse
import pickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#import imageio

# Environment
from multiagent_mujoco.mujoco_multi import MujocoMulti
from utilis import MLPNetwork



class Runner:
    def __init__(self, args):
        self.args = args
        self.seed = args.seed
        self.max_ep_len = args.max_ep_len
        self.epsilon = args.epsilon
        self.max_timesteps = args.max_timesteps
        self.policy  = []
        self.replay_buffer = []
        self.update_rule = args.update_rule
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)


        if self.args.env == "MAmujoco": # Multi-agent Mujoco
            env_args = {"scenario": self.args.scenario, "agent_conf": self.args.env_conf, "agent_obsk":  self.args.agent_obsk,"episode_limit": self.args.max_ep_len}
            self.env = MujocoMulti(env_args=env_args)
            env_info = self.env.get_env_info()
            self.args.n_agent = env_info["n_agents"]
            self.args.state_space = env_info["obs_shape"]
            self.args.action_dim = env_info["n_actions"]
            
                
        self.n_agent = self.args.n_agent
        self.action_dim = self.args.action_dim
        self.state_space = self.args.state_space
        
        if self.update_rule in ["I2Q","IDDPG","HyDDPG"]:
            self.args.n_ensemble = 1
            for i in range(self.n_agent):
                self.policy.append(I2Q_Agent(args, i))

        if self.update_rule in [ "MMQ_Quantile"]:
            for i in range(self.n_agent):
                self.policy.append(MMQ_Q_Agent(args, i))

        for i in range(self.n_agent):
            self.replay_buffer.append(ReplayBuffer(args))
        
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Wandb_init<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        self.n_ensemble = args.n_ensemble
        if self.args.use_wandb:
            print(">>>>>>>>>>> Using wandb <<<<<<<<<<<")
            if self.args.env == "MAmujoco":
                if self.args.RO_reward:
                    project_name = f"ROMAMuJoco_{self.args.scenario}{self.args.env_conf}"
                    if self.max_ep_len != 25:
                        project_name = project_name + f"_MaxStep{self.max_ep_len}"
                
                p_name = f"Agent_{self.n_agent}_updateRule_{self.args.update_rule}_ensembleN_{self.n_ensemble}_Psample{self.args.p_sample_n}_epsilon_{self.epsilon}_Seed_{self.seed}"
                if self.args.sample_from_avg:
                    p_name = p_name + "SampleAvgBound"
                if self.args.sample_from_bound:
                    p_name = p_name + "SampleLargestLowest"
                if self.args.agent_obsk > 0:
                    p_name = p_name + f"_obsk{self.args.agent_obsk}"
                if self.args.RO_reward:
                    p_name = p_name + f"_RO_RT{self.args.RO_threshold}NC{self.args.negative_const}ROp{self.args.RO_p}"
                
           
            if self.args.random_batch:
                p_name = p_name + f"_RandomBatchSize{self.args.mini_size}"
            if self.args.update_rule == "MMQ":
                p_name = p_name + f"_SampleN{self.args.p_sample_n}"
            if self.args.update_rule == "I2Q":
                p_name = p_name + f"_UpdateChoice_{self.args.update_choice}"
                p_name = p_name + f"_lambda_{self.args.lambda_}"
            if self.args.learn_reward:
                p_name = p_name + f"_LearnReward_Rdependence{self.args.reward_dependence}"
                if self.args.start_rf_timesteps < self.args.start_timesteps:
                    p_name = p_name + f"_PT_{self.args.start_timesteps - self.args.start_rf_timesteps}StepsIter{self.args.IterTrainReward_n}"
            if self.args.n_iter_train != 1:
                p_name = p_name + f"_PIter{self.args.n_iter_train}"
            if self.args.start_fm_timesteps < self.args.start_timesteps:
                p_name = p_name + f"_PreTrainFm_{self.args.start_timesteps - self.args.start_fm_timesteps}s"
            if self.args.batch_size != 100:
                p_name = p_name + f"_BatchSize{self.args.batch_size}"
            if self.args.IterOnlyCritic_n != 1:
                p_name = p_name + f"_IterOnlyCritic{self.args.IterOnlyCritic_n}"
            if self.args.start_timesteps != 20000:
                p_name = p_name + f"_Starts{self.args.start_timesteps}"
            if self.args.max_ep_len != 25:
                p_name = p_name + f"_MaxLen{self.args.max_ep_len}"

            wandb.init(config = args, project = project_name, name = p_name, job_type = "training", reinit = True)
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Wandb_init<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        print(f"The env is: {self.args.env}")
        print("The agent number is : ", self.n_agent)
        print("The state dim for one agent is", self.state_space)
        print("The update rule is : ", self.args.update_rule)
        print("The scenario is: ", self.args.scenario)

        self.total_steps = 0
        self.true_steps = 0

    
    def run_episodes(self, evaluate = False, en = 0):
        episode_reward = 0
        episode_reward_ = 0
        obs = self.env.reset()
       

        for step in range(self.max_ep_len):
        
            if self.args.use_render and evaluate == True and en == 10:
               self.env.render() #"rgb_array"
            
            a = np.zeros((args.n_agent, args.action_dim))
            if not evaluate:
                self.true_steps += 1

            if self.args.env == "MAmujoco":
                for i in range(self.n_agent):
                    a[i] = self.policy[i].select_action(np.array(obs[i]), evaluate, self.total_steps) 
                
                reward, terminated, info= self.env.step(copy.deepcopy(a))
                
                # reward_ is the RO reward we constructed ; the reward is based on the moving forward value;
                # refer to https://gymnasium.farama.org/environments/mujoco/half_cheetah/#observation-space check the reward
                if self.args.scenario in [ "HalfCheetah-v2"] and self.args.RO_reward:
                    if info["reward_run"] * 0.05  > self.args.d_threshold:
                        reward_ = 0  - self.args.negative_const
                    elif info["reward_run"] * 0.05  > self.args.RO_threshold:
                        reward_ = -5  - self.args.negative_const - self.args.RO_p
                    else:
                        reward_ = -5  - self.args.negative_const
 
                if step == self.args.max_ep_len:
                    terminated = False

                next_obs = self.env.get_obs()
                
                if not evaluate:
                    if self.args.RO_reward:
                        s_r = reward_
                    else:
                        s_r = reward

                    for i in range(self.n_agent):
                        self.replay_buffer[i].add(obs[i], a[i], s_r, next_obs[i], terminated)

            if self.args.RO_reward:
                episode_reward_ += reward_
            

            episode_reward += reward
            obs = next_obs
            if terminated:
                break

        
        if self.args.RO_reward:
            return [episode_reward, episode_reward_]
        else:
            return episode_reward

    def run(self):
        self.eval_policy()
        while self.total_steps < self.args.max_timesteps:
            if self.total_steps % 2000 == 0:
                print("The step is :", str(self.total_steps))
                print("The true steps is:", str(self.true_steps))
                self.eval_policy()
            _  = self.run_episodes(evaluate=False)
            self.total_steps += self.args.max_ep_len
            if ( self.total_steps >= min(self.args.start_fm_timesteps, self.args.start_timesteps, self.args.start_rf_timesteps) ) & ( self.total_steps % 50 == 0 ):
                for ooo in range(self.args.n_iter_train):
                    for i in range(self.n_agent):
                        self.policy[i].train(i, self.total_steps, self.replay_buffer[i], self.args.batch_size)

            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>save model <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            if self.args.save_model and ( self.total_steps in self.args.save_set ):
                save_p = f"./MAmujoco_models/{self.args.scenario}_{self.args.env_conf}/Agent_{self.n_agent}_models_Seed_{self.args.seed}"
                if self.args.random_batch:
                    save_p = save_p + f"_RandomBatchSize{self.args.mini_size}"
                if self.args.learn_reward:
                    save_p = save_p + "_LearnReward"
                    if self.args.start_rf_timesteps < self.args.start_timesteps:
                        save_p = save_p + f"_PT_{self.args.start_timesteps - self.args.start_rf_timesteps}Steps"
                if self.args.update_rule == "MMQ":
                    save_p = save_p + f"_SampleN{self.args.p_sample_n}"
                if self.args.update_rule == "I2Q" or self.args.update_rule == "IDDPG":
                    save_p = save_p + f"_Startsteps{self.args.start_timesteps}"
                if self.args.n_iter_train != 1:
                    save_p = save_p + f"_PIter{self.args.n_iter_train}"
                if self.args.start_fm_timesteps < self.args.start_timesteps:
                    save_p = save_p + f"_PreTrainFm_{self.args.start_timesteps - self.args.start_fm_timesteps}s"
                if self.args.batch_size != 100:
                    save_p = save_p + f"_BatchSize{self.args.batch_size}"

                save_p = save_p + f"/step_{self.total_steps}"
                if not os.path.exists(save_p):
                    os.makedirs(save_p)
                for i in range(self.n_agent):
                    self.policy[i].save(f"{save_p}/UpdateRule_{self.args.update_rule}_seed_{self.seed}", i)
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>save model <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    def eval_policy(self):
        reward_record = []
        reward_record_t = []
        for e_n in range(self.args.eval_episodes):
            if self.args.RO_reward:
                episode_reward, episode_reward_ = self.run_episodes(evaluate=True,en = e_n)
                reward_record.append(episode_reward_)
                reward_record_t.append(episode_reward)
            else:
                episode_reward = self.run_episodes(evaluate=True, en = e_n)
                reward_record.append(episode_reward)
        
        print("---------------------------------------")
        print(f"Evaluation over {self.args.eval_episodes} episodes Max: {np.max(reward_record):.3f}")
        print(f"Evaluation over {self.args.eval_episodes} episodes Min: {np.min(reward_record):.3f}")
        print(f"Evaluation over {self.args.eval_episodes} episodes Std: {np.std(reward_record):.3f}")
        print(f"Evaluation over {self.args.eval_episodes} episodes Avg: {np.mean(reward_record):.3f}")
        print("---------------------------------------")

        if self.args.use_wandb:
            wandb.log({'evaluation_max_return':np.max(reward_record)}, step=self.total_steps)
            wandb.log({'evaluation_min_return':np.min(reward_record)}, step=self.total_steps)
            wandb.log({'evaluation_std_return':np.std(reward_record)}, step=self.total_steps)
            wandb.log({'evaluation_avg_return':np.mean(reward_record)}, step=self.total_steps)
            
            if self.args.RO_reward:
                wandb.log({'evaluation_max_return_Default':np.max(reward_record_t)}, step=self.total_steps)
                wandb.log({'evaluation_min_return_Default':np.min(reward_record_t)}, step=self.total_steps)
                wandb.log({'evaluation_std_return_Default':np.std(reward_record_t)}, step=self.total_steps)
                wandb.log({'evaluation_avg_return_Default':np.mean(reward_record_t)}, step=self.total_steps)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # <<<<<< Basic parameters
    parser.add_argument("--buffer_size", default = 550000, type=int) # max batch capacity
    parser.add_argument("--n_agent", default = 2, type = int) # number of agents
    parser.add_argument("--state_space", default = 2, type = int) # state space
    parser.add_argument("--action_dim", default = 1, type = int) # action dim
    parser.add_argument("--max_ep_len", default = 25, type= int) # max episode length
    parser.add_argument("--seed", default=0, type=int)   # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--set_env_seed", default = True, type = bool) # set a seed for the environment dynamics
    parser.add_argument("--set_init_seed", default = True, type = bool) # set a seed for neural network initialization
    parser.add_argument("--set_train_seed", default = True, type = bool) # set a seed for action selection
    parser.add_argument("--device", default = "cpu")
    
    
    # <<<<< training parameters
    parser.add_argument("--discount", default=0.99, type = float)       # Discount factor
    parser.add_argument("--start_timesteps", default=20000, type=int) # Time steps initial random policy is used
    parser.add_argument("--max_timesteps", default=500000, type=int)   # Max time steps to run environment
    parser.add_argument("--batch_size", default=100, type=int)  # Batch size for both actor and critic
    parser.add_argument("--tau", default=0.005, type=float)   # Target network update rate
    parser.add_argument("--update_rule", default = "E") # I2Q, HyDDPG, IDDPG, MMQ_quantile
    parser.add_argument("--lr", default = 0.001, type = float) # learning rate
    parser.add_argument("--n_iter_train", default = 1, type=int)  # number of training every fixed interaction steps 
    parser.add_argument("--epsilon", default = 0.1, type = float) # Exploration
    
    # <<<<< Evaluation
    parser.add_argument("--eval_episodes", default = 20, type = int) # number of episode for Evaluation
    
    # <<<<< Environment
    parser.add_argument("--env", default = "MAMujoco")
    parser.add_argument("--scenario", default = "HalfCheetah-v2")
    parser.add_argument("--env_conf", default = "2x4") 
    parser.add_argument("--agent_obsk", default = 0, type = int) # 0 represent the observation limited its own controlled joint；1 means more. 
    parser.add_argument("--RO_reward", action = 'store_true') 
    parser.add_argument("--d_threshold", default = 0.025, type = float)
    parser.add_argument("--RO_threshold", default = 0.025, type = float)
    parser.add_argument("--negative_const", default = 0, type = float)
    parser.add_argument("--RO_p", default = 1,  type = float) # penalty
    
    # <<<<< Algorithm
    # <<< for ensemble model dynamics
    parser.add_argument("--n_ensemble", default = 1, type = int) # Ensemble number
    parser.add_argument("--random_batch",  action = 'store_true') # for different network in ensemble, whether to update each using a slightly differet batch
    parser.add_argument("--mini_size", default = 80, type = int) # random_batch, every forward model use size = 80 batch sample from size = 100 batch
    parser.add_argument("--p_sample_n", default = 1, type = int ) # number of samples drawn from the quantile for update; corresponding to M in the paper; 
    parser.add_argument("--start_fm_timesteps", default = 20000, type = int)
    parser.add_argument("--quantile_hidden_dim", default = 256, type = int)
    parser.add_argument("--sample_from_avg", action = 'store_true') 
    parser.add_argument("--sample_from_bound", action = 'store_true')

    # <<< for I2Q
    parser.add_argument("--update_choice", default = "first")
    parser.add_argument("--lambda_", default = 0.2, type = float) # regularize the update ; hyper-parameter in I2Q
    parser.add_argument("--IterOnlyCritic_n", default = 1, type = int)

    # <<< For learnd reward function
    parser.add_argument("--learn_reward",  action = 'store_true')
    parser.add_argument("--reward_dependence", default = "ns") # include next state, ns ; state, action, nextstate, sans; state, action, sa
    parser.add_argument("--start_rf_timesteps", default = 20000, type = int)
    
    # <<<<<< Visualization
    parser.add_argument("--use_wandb", action = 'store_true')
    parser.add_argument("--use_render", action = 'store_true')
    parser.add_argument("--record_one", action = 'store_true')
    
    # <<<<< save and load
    parser.add_argument("--load_model", default="")         # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--save_dir", default=".")          # OpenAI gym environment name
    parser.add_argument("--save_model", action = 'store_true') # Whether save model
    parser.add_argument("--save_set", nargs='+', type = int, default = [55000,320000])
    args = parser.parse_args()
    
    models_dir = os.path.join(args.save_dir, "models")
        
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    runner = Runner(args)
    runner.run()

    if args.use_wandb:
        wandb.finish()





