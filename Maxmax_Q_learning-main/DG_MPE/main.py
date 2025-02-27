import os, sys, time
import numpy as np
import copy
import torch
import wandb
from I2Q_model import I2Q_Agent
from MMQ_model import MMQ_Agent
from MMQ_model_Quantile import MMQ_Q_Agent
from buffer import ReplayBuffer
from env_wraf import Sequential_Target_Env
import argparse
import pickle

# Environment
from world_ns import World
from make_env import make_env # multiple particle environment
from utilis import MLPNetwork, MLPNetwork2


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

        # Environment Setting
        if self.args.env == "DG": # Differential_Games
            self.env = World(self.args)
            self.args.state_space = args.n_agent * 1
            self.args.action_dim = 1
        
        elif self.args.env == "MPE":
            ss = self.args.scenario
            if ss in [ "simple_tag", "simple_tag_RO"]:
                self.prey_agent = MLPNetwork(14, 5)
                pfile = "pretrained_agent/simple_tag_maddpg_episode_10000.pt"
                p_params = torch.load(pfile, map_location = torch.device(self.args.device))
                self.prey_agent.load_state_dict( p_params['agent_params'][3]['policy'] )
            
            if ss in ["simple_tag_RO_N2", "simple_tag_RO2_N2", "simple_tag_RO3_N2"]:
                self.prey_agent = MLPNetwork2(12, 5)
                pfile = "pretrained_agent/Steps_100000_2_actor"
                p_params = torch.load(pfile, map_location = torch.device(self.args.device))
                self.prey_agent.load_state_dict(p_params)

            self.env = make_env(self.args.scenario, self.args.n_agent)
            self.args.state_space = self.env.observation_space[0].shape[0]
            self.args.action_dim = self.env.action_space[0].n

        elif self.args.env == "Sequential_MPE":
            self.env = Sequential_Target_Env(args)
            self.args.state_space = self.env.state_space 
            self.args.action_dim = self.env.action_dim

        
        self.n_agent = self.args.n_agent
        self.action_dim = self.args.action_dim
        self.state_space = self.args.state_space
        
        # Different update rule
        if self.update_rule == "I2Q" or self.update_rule == "IDDPG" or self.update_rule == "DisDDPG" or self.update_rule == "HyDDPG":
            self.args.n_ensemble = 1
            for i in range(self.n_agent):
                self.policy.append(I2Q_Agent(args, i))
        if self.update_rule == "MMQ":
            for i in range(self.n_agent):
                self.policy.append(MMQ_Agent(args, i ))
        if self.update_rule == "MMQ_Quantile":
            for i in range(self.n_agent):
                self.policy.append(MMQ_Q_Agent(args, i))
        for i in range(self.n_agent):
            self.replay_buffer.append(ReplayBuffer(args))
        
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Wandb_init<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        self.n_ensemble = args.n_ensemble
        if self.args.use_wandb:
            print(">>>>>>>>>>> Using wandb <<<<<<<<<<<")
            if self.args.env == "DG":
                project_name = f"Paper_DG_{self.args.n_agent}Agents_RdependNS"
                p_name = f"Agent_{self.n_agent}_updateRule_{self.args.update_rule}_m_{self.args.m}_SR_{self.args.sub_reward}_epsilon_{self.epsilon}_Seed_{self.seed}"
            elif self.args.env == "MPE":
                project_name = f"MPE_{self.args.scenario}"
                p_name = f"{self.args.scenario}_Agent_{self.n_agent}_updateRule_{self.args.update_rule}_epsilon_{self.epsilon}_Seed_{self.seed}"
            elif self.args.env == "Sequential_MPE":
                project_name = "Sequential_MPE_Tasks"
                p_name = f"{self.args.scenario}_updateRule_{self.args.update_rule}_epsilon_{self.epsilon}_Seed_{self.seed}"
            if self.args.use_uniform_sample:
                p_name = p_name + f"_USstd{self.args.std_range}"
            if self.args.shift_reward:
                p_name = p_name + f"_ShiftR-{self.args.negative_const}"
            if self.args.update_rule == "I2Q":
                p_name = p_name + f"_lambda_{self.args.lambda_}"
            if self.args.update_rule in ["MMQ","MMQ_Quantile"]:
                p_name = p_name + f"_ensembleN_{self.n_ensemble}"
                p_name = p_name + f"_SampleN{self.args.p_sample_n}"
            if self.args.learn_reward:
                p_name = p_name + f"_LearnReward_Rdependence{self.args.reward_dependence}"
                if self.args.start_rf_timesteps < self.args.start_timesteps:
                    p_name = p_name + f"_PT_{self.args.start_timesteps - self.args.start_rf_timesteps}StepsIter{self.args.IterTrainReward_n}"
            if self.args.n_iter_train != 1:
                p_name = p_name + f"_PIter{self.args.n_iter_train}"
            if self.args.start_fm_timesteps < self.args.start_timesteps:
                p_name = p_name + f"_PreTrainFm_{self.args.start_timesteps - self.args.start_fm_timesteps}s"
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
        print("The state dim is :", self.args.state_space)
        print("The action dim is:", self.args.action_dim)
        print("The update rule is : ", self.args.update_rule)

        self.total_steps = 0
        self.true_steps = 0

    def run_episodes(self, evaluate = False):
        episode_reward = 0
        obs = self.env.reset()

       
        for step in range(self.max_ep_len):
            a = np.zeros((args.n_agent, args.action_dim))
            if not evaluate:
                self.true_steps += 1
            
            if self.args.env in [ "DG", "DG_noisyR", "DG_noisyS"]:
                for i in range(self.n_agent):
                    a[i] = (self.policy[i].select_action(obs, evaluate, self.total_steps))
                next_obs, reward, terminated = self.env.step(np.hstack(a))
                if self.args.shift_reward and (not evaluate):
                    reward -= self.args.negative_const
                if not evaluate:
                    for i in range(self.n_agent):
                        self.replay_buffer[i].add(obs, a[i], reward, next_obs, terminated)
                            
            elif self.args.env in [ "MPE", "Sequential_MPE"] :
                ss = self.args.scenario
                if ss in [ "simple_tag", "simple_tag_RO", "simple_tag_RO_N2", "simple_tag_RO2_N2","simple_tag_RO3_N2"]:
                    a = np.zeros((args.n_agent + 1, args.action_dim))
                    for i in range(self.n_agent):
                        a[i] = (self.policy[i].select_action(np.array(obs[i]), evaluate, self.total_steps))
                    a[i+1] = self.prey_agent(torch.FloatTensor(obs[i+1])).detach().cpu().numpy()
                else:
                    for i in range(self.n_agent):
                        a[i] = (self.policy[i].select_action(np.array(obs[i]), evaluate, self.total_steps))

                next_obs, reward, terminated, info = self.env.step(copy.deepcopy(a))
                if self.args.shift_reward and (not evaluate):
                    for o in range(self.n_agent):
                        reward[o] -= self.args.negative_const
                if not evaluate:
                    for i in range(self.n_agent):
                        self.replay_buffer[i].add(obs[i], a[i], reward[i], next_obs[i], terminated[i])
                reward = reward[0]
                terminated = terminated[0]



            episode_reward += reward
            obs = next_obs


            if terminated:
                break

  
        return episode_reward

    def run(self):
        self.eval_policy()
        while self.total_steps < self.args.max_timesteps:
            if self.total_steps % 1000 == 0:
                print("The step is :", str(self.total_steps))
                print("The true steps is:", str(self.true_steps))
                self.eval_policy()
            _  = self.run_episodes(evaluate=False)
            self.total_steps += self.args.max_ep_len
            if ( self.total_steps > min(self.args.start_fm_timesteps, self.args.start_timesteps, self.args.start_rf_timesteps) ) & ( self.total_steps % 50 == 0 ):
                for ooo in range(self.args.n_iter_train):
                    for i in range(self.n_agent):
                        self.policy[i].train(i, self.total_steps, self.replay_buffer[i], self.args.batch_size)

            if self.args.save_model and self.total_steps >= 20000 and self.total_steps % 1000 == 0:
                if self.args.env == "DG":
                    save_p = f"./models/Agent_{self.n_agent}_models_{self.args.update_rule}_m_{self.args.m}_subReward_{self.args.sub_reward}/models_Seed_{self.args.seed}_Ensemble_{self.args.n_ensemble}"
                if self.args.env == "MPE":
                    save_p = f"./models/MEP_{self.args.scenario}_{self.args.update_rule}/Seed_{self.args.seed}_Ensemble_{self.args.n_ensemble}"
                if self.args.learn_reward:
                    save_p = save_p + "_LearnReward"
                if self.args.update_rule in ["MMQ", "MMQ_Quantile"]:
                    save_p = save_p + f"_SampleN{self.args.p_sample_n}"
                if self.args.n_iter_train != 1:
                    save_p = save_p + f"_PIter{self.args.n_iter_train}"
                if self.args.batch_size != 100:
                    save_p = save_p + f"_BatchSize{self.args.batch_size}"

                save_p = save_p + f"_step_{self.total_steps}"
                if not os.path.exists(save_p):
                    os.makedirs(save_p)
                for i in range(self.n_agent):
                    self.policy[i].save(f"{save_p}/UpdateRule_{self.args.update_rule}", i)
        

    def eval_policy(self):
        reward_record = []
        w_record = []
        wnn_record = []
        for e_n in range(self.args.eval_episodes): 
        	episode_reward = self.run_episodes(evaluate=True)
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
    parser.add_argument("--update_rule", default = "E") # update rule for QSA ;zero: normal Q-learning; first ; second
    parser.add_argument("--lr", default = 0.001, type = float) # learning rate
    parser.add_argument("--n_iter_train", default = 1, type=int) # number of training for each 50 interaction step
    parser.add_argument("--shift_reward", action = 'store_true') # Reward shifting https://proceedings.neurips.cc/paper_files/paper/2022/file/f600d1a3f6a63f782680031f3ce241a7-Paper-Conference.pdf
    parser.add_argument("--negative_const", default = 0, type = float)

    # <<<<< Exploration
    parser.add_argument("--epsilon", default = 0.1, type = float) # Exploration
    
    # <<<<< Evaluation
    parser.add_argument("--eval_episodes", default = 100, type = int) # Evaluation times
    
    # <<<<< Environment
    parser.add_argument("--env", default = "DG")
    parser.add_argument("--scenario", default = "simple_spread_RO1")
    
    # Differential Game
    parser.add_argument("--m", default = 0.3, type = float)
    parser.add_argument("--sub_reward", default = 0.2, type = float)
    
    
    # <<<<< Algorithm
    # <<< for ensemble model dynamics
    parser.add_argument("--n_ensemble", default = 5, type = int) # Ensemble number
    parser.add_argument("--random_batch",  action = 'store_true')
    parser.add_argument("--mini_size", default = 80, type = int)

    # <<< for I2Q
    parser.add_argument("--update_choice", default = "first")
    parser.add_argument("--lambda_", default = 0.2, type = float) # regularize the update
    parser.add_argument("--IterQSS_n", default = 1, type = int)
    
    # <<< for Probabilistic_model P_Q
#    wparser.add_argument("--fm_use_target", default = False, type = bool)
    parser.add_argument("--p_sample_n", default = 1, type = int )
    parser.add_argument("--start_fm_timesteps", default = 20000, type = int)
    parser.add_argument("--IterOnlyCritic_n", default = 1, type = int)
    parser.add_argument("--IterTrainReward_n", default = 1, type = int)
    
    # <<< For learnd reward function
    parser.add_argument("--learn_reward",  action = 'store_true')
    parser.add_argument("--reward_dependence", default = "ns") # Choices: (1) ns: next state ; (2) sans: state, action, nextstate; (3)sa: state, action
    parser.add_argument("--start_rf_timesteps", default = 20000, type = int)
    
    # <<< For uniform sampling from the Gaussian distribution
    parser.add_argument("--use_uniform_sample", action = 'store_true')
    parser.add_argument("--std_range", default = 2, type = int)
    
    # <<< Quantile model ;  different sampling way when using ensemble (we neglect the usage of ensemble in the final result but we keep here)
    parser.add_argument("--quantile_hidden_dim", default = 256, type = int)
    parser.add_argument("--sample_from_avg", action = 'store_true') # Sampling from the avg bound of ensemble
    parser.add_argument("--sample_from_bound", action = 'store_true') # Sampling from the [ minimum value of the ensemble, maxium value of the ensemble ]

    # <<<<<< Visualization
    parser.add_argument("--use_wandb", action = 'store_true')
    parser.add_argument("--record_one", action = 'store_true')
    
    # <<<<< save and load
    parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--save_dir", default=".")          # OpenAI gym environment name
    parser.add_argument("--save_model", action = 'store_true') # Whether save model
    parser.add_argument("--save_set", nargs='+', type = int, default = [100000,200000,300000,500000])
   


    args = parser.parse_args()
    

#    results_dir = os.path.join(args.save_dir, "results")
    models_dir = os.path.join(args.save_dir, "models")
        
#    if not os.path.exists(results_dir):
#        os.makedirs(results_dir)

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    runner = Runner(args)
    runner.run()

    if args.use_wandb:
        wandb.finish()





