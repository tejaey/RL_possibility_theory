
import torch
import copy
import torch.nn as nn
import numpy as np
import random
import wandb
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
    
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.constant_(self.fc3.bias, 0)
    
    def forward(self, state):
        a = F.relu(self.fc1(state))
        a = F.relu(self.fc2(a))
        a = torch.tanh(self.fc3(a))

        return a

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
    
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.constant_(self.fc3.bias, 0)

    def forward(self, state, action):
        q = torch.cat([state,action], dim = -1)
        q = F.relu(self.fc1(q))
        q = F.relu(self.fc2(q))
        q = self.fc3(q)

        return q

class P_Forward_model_DG(nn.Module):
    def __init__(self, state_dim, action_dim, init_type = "xavier", zero_bias = True):
        super(P_Forward_model_DG, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, state_dim)
        self.fc4 = nn.Linear(256, state_dim)
        
        if init_type == "xavier":
            nn.init.xavier_uniform_(self.fc1.weight)
            nn.init.xavier_uniform_(self.fc2.weight)
            nn.init.xavier_uniform_(self.fc3.weight)
            nn.init.xavier_uniform_(self.fc4.weight)
        if zero_bias:
            nn.init.constant_(self.fc1.bias, 0)
            nn.init.constant_(self.fc2.bias, 0)
            nn.init.constant_(self.fc3.bias, 0)
            nn.init.constant_(self.fc4.bias, 0)
    
    def reparameterize(self, mean, log_var):
        noi = torch.randn_like(mean)
        std = torch.exp(0.5*log_var)
        z = mean + noi * std
        z = torch.clamp(z, -1, 1) # torch.clamp could clamp each item in z between -1 and 1

        return z

    def forward(self, state, action):
        
        s  = torch.cat([state, action], dim=-1)
        s = F.relu(self.fc1(s))
        s = F.relu(self.fc2(s))
        
        ds = torch.tanh(self.fc3(s))
        mu = 0.1 * ds + state
        mu = torch.clamp(mu, -1, 1)
        logvar = self.fc4(s)
        sample_z = self.reparameterize(mu, logvar)
        
        return mu, logvar, sample_z

class P_Forward_model(nn.Module):
    def __init__(self, state_dim, action_dim, init_type = "xavier", zero_bias = True):
        super(P_Forward_model, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, state_dim)
        self.fc4 = nn.Linear(256, state_dim)
        
        if init_type == "xavier":
            nn.init.xavier_uniform_(self.fc1.weight)
            nn.init.xavier_uniform_(self.fc2.weight)
            nn.init.xavier_uniform_(self.fc3.weight)
            nn.init.xavier_uniform_(self.fc4.weight)
        if zero_bias:
            nn.init.constant_(self.fc1.bias, 0)
            nn.init.constant_(self.fc2.bias, 0)
            nn.init.constant_(self.fc3.bias, 0)
            nn.init.constant_(self.fc4.bias, 0)
    
    def reparameterize(self, mean, log_var):
        noi = torch.randn_like(mean)
        std = torch.exp(0.5*log_var)
        z = mean + noi * std

        return z
    
    def forward(self, state, action):
        
        s  = torch.cat([state, action], dim=-1)
        s = F.relu(self.fc1(s))
        s = F.relu(self.fc2(s))
        
        ds = self.fc3(s)
        mu = ds + state
        logvar = self.fc4(s)
        sample_z = self.reparameterize(mu, logvar)
        
        return mu, logvar, sample_z

class P_Forward_model_uniform_DG(nn.Module):
    def __init__(self, state_dim, action_dim, std_range,  init_type = "xavier", zero_bias = True):
        super(P_Forward_model_uniform_DG, self).__init__()
        self.std_range =  std_range
        self.state_dim =  state_dim
        
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, state_dim)
        self.fc4 = nn.Linear(256, state_dim)
        
        if init_type == "xavier":
            nn.init.xavier_uniform_(self.fc1.weight)
            nn.init.xavier_uniform_(self.fc2.weight)
            nn.init.xavier_uniform_(self.fc3.weight)
            nn.init.xavier_uniform_(self.fc4.weight)
        if zero_bias:
            nn.init.constant_(self.fc1.bias, 0)
            nn.init.constant_(self.fc2.bias, 0)
            nn.init.constant_(self.fc3.bias, 0)
            nn.init.constant_(self.fc4.bias, 0)
    
    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5*log_var)
        low = mean - self.std_range * std  # lower and higher bound
        high = mean + self.std_range * std

        z = torch.FloatTensor( np.random.uniform(low.detach().cpu(), high.detach().cpu(), size = ( low.shape[0], self.state_dim)) ).to(device)
        z = torch.clamp(z, -1, 1) # torch.clamp could clamp each item in z between -1 and 1
        
        return z
    
    def forward(self, state, action):
        
        s  = torch.cat([state, action], dim=-1)
        s = F.relu(self.fc1(s))
        s = F.relu(self.fc2(s))
        
        ds = torch.tanh(self.fc3(s))
        mu = 0.1 * ds + state
        mu = torch.clamp(mu, -1, 1)
        logvar = self.fc4(s)
        sample_z = self.reparameterize(mu, logvar)

        return mu, logvar, sample_z

class P_Forward_model_uniform(nn.Module):
    def __init__(self, state_dim, action_dim, std_range,  init_type = "xavier", zero_bias = True):
        super(P_Forward_model_uniform, self).__init__()
        
        self.std_range = std_range
        self.state_dim = state_dim
        
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, state_dim)
        self.fc4 = nn.Linear(256, state_dim)
        
        if init_type == "xavier":
            nn.init.xavier_uniform_(self.fc1.weight)
            nn.init.xavier_uniform_(self.fc2.weight)
            nn.init.xavier_uniform_(self.fc3.weight)
            nn.init.xavier_uniform_(self.fc4.weight)
        if zero_bias:
            nn.init.constant_(self.fc1.bias, 0)
            nn.init.constant_(self.fc2.bias, 0)
            nn.init.constant_(self.fc3.bias, 0)
            nn.init.constant_(self.fc4.bias, 0)
    
    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5*log_var)
        low = mean - self.std_range * std  # lower and higher bound
        high = mean + self.std_range * std
        z = torch.FloatTensor( np.random.uniform(low.detach().cpu(), high.detach().cpu(), size = ( low.shape[0], self.state_dim)) ).to(device)
        
        return z
    
    def forward(self, state, action):
        
        s  = torch.cat([state, action], dim=-1)
        s = F.relu(self.fc1(s))
        s = F.relu(self.fc2(s))
        
        ds = self.fc3(s)
        mu = ds + state
        logvar = self.fc4(s)
        sample_z = self.reparameterize(mu, logvar)
        
        return mu, logvar, sample_z

class Reward_function(nn.Module):
    def __init__(self, input_dim):
        super(Reward_function, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.constant_(self.fc3.bias, 0)
    
    def forward(self, input):
#        r = torch.cat([state,action], dim = -1)
        r = F.relu(self.fc1(input))
        r = F.relu(self.fc2(r))
        r = self.fc3(r)
        
        return r

class MMQ_Agent():
    def __init__(self, args, i = None): #lr = 3e-4
        self.args = args
        self.epsilon = args.epsilon
        self.lr = args.lr
        self.tau = args.tau
        self.discount = args.discount
        self.n_ensemble = args.n_ensemble
        
#        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(self.args.device)
            
        if self.args.set_init_seed:
            np.random.seed(self.args.seed )
            torch.manual_seed(self.args.seed )

        self.actor = Actor(args.state_space, args.action_dim).to(self.device)
        self.target_actor = copy.deepcopy(self.actor)
        
        self.critic = Critic(args.state_space, args.action_dim).to(self.device)
        self.target_critic = copy.deepcopy(self.critic)

        self.forward_model = []
    
        for i in range(self.n_ensemble):
            if self.args.env == "DG":
                if self.args.use_uniform_sample:
                    self.forward_model.append(P_Forward_model_uniform_DG(args.state_space, args.action_dim, args.std_range).to(self.device))
                else:
                    self.forward_model.append(P_Forward_model_DG(args.state_space, args.action_dim).to(self.device))
            elif self.args.env == "MPE":
                if self.args.use_uniform_sample:
                    self.forward_model.append(P_Forward_model_uniform(args.state_space, args.action_dim, args.std_range).to(self.device))
                else:
                    self.forward_model.append(P_Forward_model(args.state_space, args.action_dim).to(self.device))

        self.actor_optimizer = optim.Adam(self.actor.parameters(), self.lr)
        self.critic_optimizer = []
        self.critic_optimizer = optim.Adam(self.critic.parameters(), self.lr)

        
        self.forward_model_optimizer = []
        for i in range(self.n_ensemble):
            self.forward_model_optimizer.append(optim.Adam(self.forward_model[i].parameters(), self.lr))


        self.total_it = 0
        
        if self.args.learn_reward:
            if self.args.reward_dependence == "ns" or self.args.reward_dependence == "s":
                input_dim =self.args.state_space
            elif self.args.reward_dependence == "sans":
                input_dim = self.args.state_space * 2 + self.args.action_dim
            elif self.args.reward_dependence == "sa":
                input_dim = self.args.state_space + self.args.action_dim

            self.reward_function = Reward_function(input_dim).to(self.device)
            self.reward_function_optimizer = optim.Adam(self.reward_function.parameters(),self.lr)

        if self.args.set_train_seed:
            np.random.seed(self.args.seed )
            torch.manual_seed(self.args.seed )


    def select_action(self, state, evaluate, total_step):
        state = torch.FloatTensor(state).to(self.device)
        epsilon = 0 if evaluate else self.args.epsilon
        with torch.no_grad():
            if not evaluate:
                if total_step < self.args.start_timesteps:
                    action = 2 * np.random.rand(self.args.action_dim) - 1
                elif (np.random.uniform() < epsilon):
                    action = 2 * np.random.rand(self.args.action_dim) - 1
                else:
                    action = np.clip( self.actor(state).cpu().data.numpy() + 0.1*np.random.randn(self.args.action_dim), -1,1)
            else:
                action = self.actor(state).cpu().data.numpy()

        return action
                
    
    def train(self, id, total_steps, buffer, batch_size = 100):
        
        states, actions, rewards, next_states, dones = buffer.getBatch(batch_size)
        self.total_it += 1
        
        for ee in range(self.n_ensemble):
            mu, logvar, _  = self.forward_model[ee](states, actions)
            

        if  ( total_steps > min( self.args.start_fm_timesteps, self.args.start_timesteps) ):
            for ee in range(self.n_ensemble):
                if self.args.random_batch:
                    mini_index = np.random.choice(np.arange(batch_size), size = self.args.mini_size, replace=False)
                    mu, logvar, _ = self.forward_model[ee](states[mini_index], actions[mini_index])
                    std = torch.sqrt(torch.exp(logvar))
                    forward_model_loss = torch.mean( (next_states[mini_index] - mu ) ** 2 / (2 * (std)**2) + (1/2) * torch.log((std)**2))
                    mu_diff = F.mse_loss(mu, next_states[mini_index]) # use for record
                else:
                    mu, logvar, _ = self.forward_model[ee](states, actions)
                    std = torch.sqrt(torch.exp(logvar))
                    forward_model_loss =  torch.mean( (next_states - mu ) ** 2 / (2 * (std)**2) + (1/2) * torch.log((std)**2))
                    mu_diff = F.mse_loss(mu, next_states) # use for record


                # optimize the forward model
                self.forward_model_optimizer[ee].zero_grad()
                forward_model_loss.backward()
                self.forward_model_optimizer[ee].step()

                
        if ( total_steps > min( self.args.start_rf_timesteps, self.args.start_timesteps) ) :
            if self.args.learn_reward:
            #<<<<< Reward function
                if self.args.reward_dependence == "ns":
                    input_data = next_states
                elif self.args.reward_dependence == "s":
                    input_data = states
                elif self.args.reward_dependence == "sans":
                    input_data = torch.cat([states,actions,next_states], dim = -1)
                elif self.args.reward_dependence == "sa":
                    input_data = torch.cat([states,actions], dim = -1)

                for ttt in range(self.args.IterTrainReward_n):
                    reward_function_loss = F.mse_loss(self.reward_function(input_data), rewards)
                    self.reward_function_optimizer.zero_grad()
                    reward_function_loss.backward()
                    self.reward_function_optimizer.step()

        if ( total_steps > self.args.start_timesteps):
            actor_loss = - self.critic(states, self.actor(states)).mean()
            # Optimize the critic
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
    
            all_predict_mu = torch.zeros((batch_size, self.args.state_space, self.n_ensemble))
            all_predict_logvar = torch.zeros((batch_size, self.args.state_space, self.n_ensemble))
            all_predict_Q = torch.zeros((batch_size, self.n_ensemble * self.args.p_sample_n + 1))
            all_predict_backup = torch.zeros((batch_size, self.n_ensemble * self.args.p_sample_n + 1))
            all_predict_states = torch.zeros((batch_size, self.args.state_space, self.n_ensemble * self.args.p_sample_n + 1))
            
            used_next_states = copy.deepcopy(next_states)
            for ee in range(self.n_ensemble):
                mu, logvar, _  = self.forward_model[ee](states, actions)
                # For record
                all_predict_mu[:, :,  ee] = copy.deepcopy(mu.detach())
                all_predict_logvar[:, :, ee] = copy.deepcopy(logvar.detach())
                for ppp in range(self.args.p_sample_n):
                    _, _, ps = self.forward_model[ee](states, actions)
                    all_predict_states[:, :, ee * self.args.p_sample_n + ppp] = copy.deepcopy(ps.detach())
                    if self.args.learn_reward:
                        if self.args.reward_dependence == "ns":
                            ps_data = ps
                            ps_reward = self.reward_function(ps_data).detach() # size = [batch_size,1]
                        elif self.args.reward_dependence == "sans":
                            ps_data = torch.cat([states, actions, ps], dim = -1)
                            ps_reward = self.reward_function(ps_data).detach() # size = [batch_size,1]
                    else:
                        ps_reward = rewards

                    all_predict_Q[:, ee * self.args.p_sample_n + ppp] = self.target_critic(ps, self.target_actor(ps).detach()).squeeze(-1)
                    backup = (ps_reward + (1-dones) * self.discount * self.target_critic(ps, self.target_actor(ps).detach()) ).squeeze(-1)
                    #size = [ batch_size]
                    all_predict_backup[:, ee * self.args.p_sample_n + ppp ] = copy.deepcopy(backup.detach())
        
            next_Q  = self.target_critic(next_states, self.target_actor(next_states)).detach().squeeze(-1)
            all_predict_Q[:, self.args.n_ensemble * self.args.p_sample_n ] = copy.deepcopy(next_Q)
            all_predict_backup[:, self.args.n_ensemble * self.args.p_sample_n ] = ( rewards + (1-dones) * self.discount * copy.deepcopy(next_Q.unsqueeze(-1))).squeeze(-1)
            all_predict_states[:, :, self.args.n_ensemble * self.args.p_sample_n] = copy.deepcopy(next_states.detach().cpu())


            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< back-up value >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            max_index = np.argmax(all_predict_backup, axis = -1) # size = batch_size
            max_state = all_predict_states[np.arange(batch_size),:, max_index] # size = [ batch_size, state_space ]
            max_backup = torch.FloatTensor(all_predict_backup[np.arange(batch_size), max_index]).unsqueeze(-1).to(self.device) # size = [batch_size, 1]

            for ccc in range(self.args.IterOnlyCritic_n):
                critic_loss= F.mse_loss(self.critic(states, actions), max_backup)
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()
            
                    
            
            #<<<<< Target network
            for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return

    

    def save(self, filename, agent_idx):
        torch.save(self.critic.state_dict(), filename + "_" + str(agent_idx) + "_critic")
        torch.save(self.target_critic.state_dict(), filename + "_" + str(agent_idx) + "_target_critic")
        torch.save(self.actor.state_dict(), filename + "_" + str(agent_idx) + "_actor")
        torch.save(self.target_actor.state_dict(), filename + "_" + str(agent_idx) + "_target_actor")
        for j in range(self.args.n_ensemble):
            torch.save(self.forward_model[j].state_dict(), filename + "_" + str(agent_idx) + f"_forward_model_{j}")
        if self.args.learn_reward:
            torch.save(self.reward_function.state_dict(), filename + "_" + str(agent_idx) + "_RewardFunction")

    def load(self, filename, agent_idx):
        self.critic.load_state_dict(torch.load(filename + "_" + str(agent_idx) + "_critic"))
        self.target_critic.load_state_dict(torch.load(filename + "_" + str(agent_idx) + "_target_critic"))
        self.actor.load_state_dict(torch.load(filename + "_" + str(agent_idx) + "_actor"))
        self.target_actor.load_state_dict(torch.load(filename + "_" + str(agent_idx) + "_target_actor"))
        for j in range(self.args.n_ensemble):
            self.forward_model[j].load_state_dict(torch.load(filename + "_" + str(agent_idx) + f"_forward_model_{j}"))
        if self.args.learn_reward:
            self.reward_function.load_state_dict(torch.load( filename + "_" + str(agent_idx) + "_RewardFunction"))




