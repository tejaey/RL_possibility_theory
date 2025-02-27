import torch
import copy
import torch.nn as nn
import numpy as np
import wandb
import random
import torch.nn.functional as F
import torch.optim as optim

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

class Forward_model(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Forward_model, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, state_dim)
        
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.constant_(self.fc3.bias, 0)
    
    
    def forward(self, state, action):
        
        s  = torch.cat([state, action], dim=-1)
        s = F.relu(self.fc1(s))
        s = F.relu(self.fc2(s))
        ds = self.fc3(s)
        s = ds + state
    
        return s

class QSS(nn.Module):
    def __init__(self, state_dim):
        super(QSS, self).__init__()
        self.fc1 = nn.Linear(2 * state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
    
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.constant_(self.fc3.bias, 0)


    def forward(self, state, next_state):
        qss = torch.cat([state, next_state], dim = -1)
        qss = F.relu(self.fc1(qss))
        qss = F.relu(self.fc2(qss))
        qss = self.fc3(qss)

        return qss


class I2Q_Agent():
    def __init__(self, args, id): #lr = 3e-4
        self.args = args
        self.lambda_ = args.lambda_
        self.tau = args.tau
        self.discount = args.discount
        self.id = id
        
        self.device = torch.device(args.device)

        if self.args.set_init_seed:
#            random.seed(self.args.seed )
            np.random.seed(self.args.seed )
            torch.manual_seed(self.args.seed )
        
        self.actor = Actor(args.state_space, args.action_dim).to(self.device)
        self.target_actor = copy.deepcopy(self.actor)
        
        self.critic = Critic(args.state_space, args.action_dim).to(self.device)
        self.target_critic = copy.deepcopy(self.critic)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(),args.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(),args.lr)

        if self.args.update_rule == "I2Q":
            # QSS value
            self.QSS = QSS(args.state_space).to(self.device)
            self.target_QSS = copy.deepcopy(self.QSS)

            if self.args.env == "MAmujoco":
                self.forward_model = Forward_model(args.state_space, args.action_dim).to(self.device)
            self.target_forward_model = copy.deepcopy(self.forward_model)


            self.QSS_optimizer = optim.Adam(self.QSS.parameters(),args.lr)
            self.forward_model_optimizer = optim.Adam(self.forward_model.parameters(),args.lr)

        self.total_it = 0

        if self.args.set_train_seed:
#            random.seed(self.args.seed  )
            np.random.seed(self.args.seed  )
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
    
    def train(self, id, total_steps, buffer, batch_size=100):
        

        states, actions, rewards, next_states, dones = buffer.getBatch(batch_size)
        self.total_it += 1
    
        if self.args.update_rule == "I2Q":
            #<<<< forward model
            next_predict_states = self.forward_model(states, actions)
            use_lambda = self.lambda_/torch.abs(self.QSS(states, next_predict_states)).mean().detach()
            forward_model_loss = - use_lambda * self.QSS(states, next_predict_states).mean() + F.mse_loss(next_predict_states, next_states).mean()
            model_mseloss = F.mse_loss(next_predict_states, next_states).mean()
        

            # optimize the forward model
            self.forward_model_optimizer.zero_grad()
            forward_model_loss.backward()
            self.forward_model_optimizer.step()

            #<<<<< Critic 第一种update方法
            if self.args.update_choice == "first": # Q(s,s') update Q(s,s'), r(s')
                next_predict_states = self.target_forward_model(states, actions)
                value = torch.max(self.target_QSS(states, next_predict_states), self.target_QSS(states, next_states)).detach()
                cc = ( self.target_QSS(states, next_states) < self.target_QSS(states, next_predict_states)).type(torch.float32)
            
                for i in range(self.args.IterOnlyCritic_n):
                    critic_loss = F.mse_loss(self.critic(states, actions), value)
                    # optimize the critic network
                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    self.critic_optimizer.step()

            elif self.args.update_choice == "second":
                next_predict_states = self.target_forward_model(states, actions)
                cc = (self.target_QSS(states, next_states) < self.target_QSS(states, next_predict_states)).type(torch.float32)
                use_true_states = cc * next_predict_states + (1-cc) * next_states
                Q_target1 = rewards + (1-dones) * self.discount * self.target_critic(use_true_states, self.target_actor(use_true_states))
                Q_target2 = rewards + (1-dones) * self.discount * self.target_critic(next_states, self.target_actor(next_states))
                use_target = torch.min(Q_target1, Q_target2).detach()
                QSS_loss = F.mse_loss(self.QSS(states, next_states), Q_target2)
                cc_m = torch.sum( ( Q_target1 < Q_target2).type(torch.float32))

                for i in range(self.args.IterOnlyCritic_n):
                    critic_loss = F.mse_loss(self.critic(states, actions), use_target)
                    # optimize the critic network
                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    self.critic_optimizer.step()

            else:
                next_predict_states = self.target_forward_model(states, actions)
                cc = (self.target_QSS(states, next_states) < self.target_QSS(states, next_predict_states)).type(torch.float32)
                use_true_states = cc * next_predict_states + (1-cc) * next_states
                y_q = rewards + (1-dones) * self.discount * self.target_critic(use_true_states, self.target_actor(use_true_states)).detach()
                for i in range(self.args.IterOnlyCritic_n):
                    critic_loss = F.mse_loss(self.critic(states, actions), y_q)
                    # optimize the critic network
                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    self.critic_optimizer.step()

            if self.args.update_choice != "second":
                best_indiv_action = self.target_actor(next_states)
                nextnext_predict_states_target = self.target_forward_model(next_states, best_indiv_action)
                y = rewards + (1-dones) * self.discount * self.target_QSS(next_states, nextnext_predict_states_target).detach()
                QSS_loss = F.mse_loss(self.QSS(states, next_states), y)
            
            # optimize the Qss network
            self.QSS_optimizer.zero_grad()
            QSS_loss.backward()
            self.QSS_optimizer.step()

        if self.args.update_rule == "IDDPG":
            Q_target = ( rewards + (1-dones) * self.discount * self.target_critic(next_states,self.target_actor(next_states))).detach()
            for ccc in range(self.args.IterOnlyCritic_n):
                critic_loss = F.mse_loss(self.critic(states, actions), Q_target)
                # optimize the critic network
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()
        
        if self.args.update_rule == "DisDDPG":
            Q_target = ( rewards + (1-dones) * self.discount * self.target_critic(next_states,self.target_actor(next_states))).detach()
            for ccc in range(self.args.IterOnlyCritic_n):
                loss = ( Q_target - self.critic(states, actions) )
                loss[loss<0] = 0
                critic_loss = torch.mean(loss ** 2)

                # optimize the critic network
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

        if self.args.update_rule == "HyDDPG":
            Q_target = ( rewards + (1-dones) * self.discount * self.target_critic(next_states,self.target_actor(next_states))).detach()
            for ccc in range(self.args.IterOnlyCritic_n):
                loss = ( Q_target - self.critic(states, actions) )
                loss[loss<0] = 0.5 * loss[loss<0]
                critic_loss = torch.mean(loss ** 2)

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

        #<<<<< Actor
        actor_loss = - self.critic(states, self.actor(states)).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        #<<<<< Target network
        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        if self.args.update_rule == "I2Q":
            for param, target_param in zip(self.QSS.parameters(), self.target_QSS.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.forward_model.parameters(), self.target_forward_model.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return


    def save(self, filename, agent_idx):
        torch.save(self.critic.state_dict(), filename + "_" + str(agent_idx) + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_" + str(agent_idx) + "_critic_optimizer")
        torch.save(self.actor.state_dict(), filename + "_" + str(agent_idx) + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_" + str(agent_idx) + "_actor_optimizer")
        if self.args.update_rule == "I2Q":
            torch.save(self.QSS.state_dict(), filename + "_" + str(agent_idx) + "_QSS")
            torch.save(self.QSS_optimizer.state_dict(), filename + "_" + str(agent_idx) + "_QSS_optimizer")
            torch.save(self.forward_model.state_dict(), filename + "_" + str(agent_idx) + "_forward_model")
            torch.save(self.forward_model_optimizer.state_dict(), filename + "_" + str(agent_idx) + "_forward_model_optimizer")


    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        if self.args.update_rule == "I2Q":
            self.QSS.load_state_dict(torch.load(filename + "_QSS"))
            self.QSS_optimizer.load_state_dict(torch.load(filename + "_QSS_optimizer"))

    def init_update(self):
        self.target_forward_model.load_state_dict(self.forward_model.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_actor.load_state_dict(self.actor.state_dict())
        if self.args.update_rule == "I2Q":
            self.target_QSS.load_state_dict(self.QSS.state_dict())








