import torch
import copy
import torch.nn as nn
import numpy as np
import random
import wandb
import torch.nn.functional as F
import torch.optim as optim

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        q = torch.cat([state, action], dim=-1)
        q = F.relu(self.fc1(q))
        q = F.relu(self.fc2(q))
        q = self.fc3(q)

        return q


class Quantile_model_DG(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        init_type="xavier",
        zero_bias=True,
        Q_hidden_dim=256,
    ):
        super(Quantile_model_DG, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, Q_hidden_dim)
        self.fc2 = nn.Linear(Q_hidden_dim, Q_hidden_dim)
        self.fc3 = nn.Linear(Q_hidden_dim, state_dim)
        self.fc4 = nn.Linear(Q_hidden_dim, state_dim)

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

    def forward(self, state, action):
        s = torch.cat([state, action], dim=-1)
        s = F.relu(self.fc1(s))
        s = F.relu(self.fc2(s))
        # predict the increment
        ds = self.fc3(s)
        x = ds + state
        x = torch.clamp(x, -1, 1)

        return x


class Quantile_model(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        init_type="xavier",
        zero_bias=True,
        Q_hidden_dim=256,
    ):
        super(Quantile_model, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, Q_hidden_dim)
        self.fc2 = nn.Linear(Q_hidden_dim, Q_hidden_dim)
        self.fc3 = nn.Linear(Q_hidden_dim, state_dim)
        # self.fc4 = nn.Linear(256, state_dim)

        if init_type == "xavier":
            nn.init.xavier_uniform_(self.fc1.weight)
            nn.init.xavier_uniform_(self.fc2.weight)
            nn.init.xavier_uniform_(self.fc3.weight)
            # nn.init.xavier_uniform_(self.fc4.weight)
        if zero_bias:
            nn.init.constant_(self.fc1.bias, 0)
            nn.init.constant_(self.fc2.bias, 0)
            nn.init.constant_(self.fc3.bias, 0)
            # nn.init.constant_(self.fc4.bias, 0)

    def forward(self, state, action):
        s = torch.cat([state, action], dim=-1)
        s = F.relu(self.fc1(s))
        s = F.relu(self.fc2(s))
        # predict the increment
        ds = self.fc3(s)
        x = ds + state

        return x


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


class MMQ_Q_Agent:
    def __init__(self, args, id=None):  # lr = 3e-4
        self.args = args
        self.epsilon = args.epsilon
        self.lr = args.lr
        self.tau = args.tau
        self.discount = args.discount
        self.n_ensemble = args.n_ensemble

        self.device = torch.device(self.args.device)

        if self.args.set_init_seed:
            np.random.seed(self.args.seed + id)
            torch.manual_seed(self.args.seed + id)

        self.actor = Actor(args.state_space, args.action_dim).to(self.device)
        self.target_actor = copy.deepcopy(self.actor)

        self.critic = Critic(args.state_space, args.action_dim).to(self.device)
        self.target_critic = copy.deepcopy(self.critic)

        self.forward_model = []
        # for quantile model
        self.forward_model_l = []
        self.forward_model_u = []
        for i in range(self.n_ensemble):
            if self.args.env in ["DG"]:
                self.forward_model_l.append(
                    Quantile_model_DG(
                        args.state_space,
                        args.action_dim,
                        Q_hidden_dim=args.quantile_hidden_dim,
                    ).to(self.device)
                )
                self.forward_model_u.append(
                    Quantile_model_DG(
                        args.state_space,
                        args.action_dim,
                        Q_hidden_dim=args.quantile_hidden_dim,
                    ).to(self.device)
                )

            elif self.args.env in ["MPE", "Sequential_MPE"]:
                self.forward_model_l.append(
                    Quantile_model(
                        args.state_space,
                        args.action_dim,
                        Q_hidden_dim=args.quantile_hidden_dim,
                    ).to(self.device)
                )
                self.forward_model_u.append(
                    Quantile_model(
                        args.state_space,
                        args.action_dim,
                        Q_hidden_dim=args.quantile_hidden_dim,
                    ).to(self.device)
                )

        self.actor_optimizer = optim.Adam(self.actor.parameters(), self.lr)
        self.critic_optimizer = []
        self.critic_optimizer = optim.Adam(self.critic.parameters(), self.lr)

        self.forward_model_optimizer_l = []
        self.forward_model_optimizer_u = []
        for i in range(self.n_ensemble):
            self.forward_model_optimizer_l.append(
                optim.Adam(self.forward_model_l[i].parameters(), self.lr)
            )
            self.forward_model_optimizer_u.append(
                optim.Adam(self.forward_model_u[i].parameters(), self.lr)
            )

        self.total_it = 0

        if self.args.learn_reward:
            if (
                self.args.reward_dependence == "ns"
                or self.args.reward_dependence == "s"
            ):
                input_dim = self.args.state_space
            elif self.args.reward_dependence == "sans":
                input_dim = self.args.state_space * 2 + self.args.action_dim
            elif self.args.reward_dependence == "sa":
                input_dim = self.args.state_space + self.args.action_dim

            self.reward_function = Reward_function(input_dim).to(self.device)
            self.reward_function_optimizer = optim.Adam(
                self.reward_function.parameters(), self.lr
            )

        if self.args.set_train_seed:
            np.random.seed(self.args.seed + id)
            torch.manual_seed(self.args.seed + id)

    def select_action(self, state, evaluate, total_step):
        state = torch.FloatTensor(state).to(self.device)
        epsilon = 0 if evaluate else self.args.epsilon
        with torch.no_grad():
            if not evaluate:
                if total_step < self.args.start_timesteps:
                    action = 2 * np.random.rand(self.args.action_dim) - 1
                elif np.random.uniform() < epsilon:
                    action = 2 * np.random.rand(self.args.action_dim) - 1
                else:
                    action = np.clip(
                        self.actor(state).cpu().data.numpy()
                        + 0.1 * np.random.randn(self.args.action_dim),
                        -1,
                        1,
                    )
            else:
                action = self.actor(state).cpu().data.numpy()

        return action

    def quantile_loss(self, predictions, targets, tau):
        errors = targets - predictions
        return torch.max((tau - 1) * errors, tau * errors).mean()

    def sample_ns(self, ls, us):
        z = torch.FloatTensor(
            np.random.uniform(
                ls.detach().cpu(),
                us.detach().cpu(),
                size=(ls.shape[0], self.args.state_space),
            )
        ).to(self.device)
        if self.args.env == "DG":
            z = torch.clamp(z, -1, 1)

        return z

    def train(self, id, total_steps, buffer, batch_size=100):
        states, actions, rewards, next_states, dones = buffer.getBatch(batch_size)
        self.total_it += 1

        # (1) first predict upper and lower quantile value of the state ; (2) then sampling from the upper and lower bound
        if total_steps > min(self.args.start_fm_timesteps, self.args.start_timesteps):
            for ee in range(self.n_ensemble):
                if self.args.random_batch:
                    mini_index = np.random.choice(
                        np.arange(batch_size), size=self.args.mini_size, replace=False
                    )
                    L_ns = self.forward_model_l[ee](
                        states[mini_index], actions[mini_index]
                    )
                    U_ns = self.forward_model_u[ee](
                        states[mini_index], actions[mini_index]
                    )
                    forward_model_loss_l = self.quantile_loss(
                        L_ns, next_states[mini_index], tau=0.05
                    )
                    forward_model_loss_u = self.quantile_loss(
                        U_ns, next_states[mini_index], tau=0.95
                    )
                else:
                    L_ns = self.forward_model_l[ee](states, actions)
                    U_ns = self.forward_model_u[ee](states, actions)
                    forward_model_loss_l = self.quantile_loss(
                        L_ns, next_states, tau=0.05
                    )
                    forward_model_loss_u = self.quantile_loss(
                        U_ns, next_states, tau=0.95
                    )

                # optimize the forward model
                self.forward_model_optimizer_l[ee].zero_grad()
                forward_model_loss_l.backward()
                self.forward_model_optimizer_l[ee].step()

                self.forward_model_optimizer_u[ee].zero_grad()
                forward_model_loss_u.backward()
                self.forward_model_optimizer_u[ee].step()

        if total_steps > min(self.args.start_rf_timesteps, self.args.start_timesteps):
            if self.args.learn_reward:
                # <<<<< Reward function
                if self.args.reward_dependence == "ns":
                    input_data = next_states
                elif self.args.reward_dependence == "s":
                    input_data = states
                elif self.args.reward_dependence == "sans":
                    input_data = torch.cat([states, actions, next_states], dim=-1)
                elif self.args.reward_dependence == "sa":
                    input_data = torch.cat([states, actions], dim=-1)

                for ttt in range(self.args.IterTrainReward_n):
                    reward_function_loss = F.mse_loss(
                        self.reward_function(input_data), rewards
                    )
                    self.reward_function_optimizer.zero_grad()
                    reward_function_loss.backward()
                    self.reward_function_optimizer.step()

        if total_steps > self.args.start_timesteps:
            actor_loss = -self.critic(states, self.actor(states)).mean()
            # Optimize the critic
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            all_predict_US = torch.zeros(
                (batch_size, self.args.state_space, self.n_ensemble)
            )
            all_predict_LS = torch.zeros(
                (batch_size, self.args.state_space, self.n_ensemble)
            )
            all_predict_Q = torch.zeros(
                (batch_size, self.n_ensemble * self.args.p_sample_n + 1)
            )
            all_predict_backup = torch.zeros(
                (batch_size, self.n_ensemble * self.args.p_sample_n + 1)
            )
            all_predict_states = torch.zeros(
                (
                    batch_size,
                    self.args.state_space,
                    self.n_ensemble * self.args.p_sample_n + 1,
                )
            )
            used_next_states = copy.deepcopy(next_states)

            # predict upper and lower limit, sampling K next possible state from it
            if (not self.args.sample_from_avg) and (not self.args.sample_from_bound):
                for ee in range(self.n_ensemble):
                    pred_ls = self.forward_model_l[ee](states, actions)
                    pred_us = self.forward_model_u[ee](states, actions)
                    # For record
                    all_predict_US[:, :, ee] = copy.deepcopy(pred_us.detach())
                    all_predict_LS[:, :, ee] = copy.deepcopy(pred_ls.detach())
                    for ppp in range(self.args.p_sample_n):
                        ps = self.sample_ns(pred_ls, pred_us)
                        all_predict_states[:, :, ee * self.args.p_sample_n + ppp] = (
                            copy.deepcopy(ps.detach())
                        )
                        if self.args.learn_reward:
                            if self.args.reward_dependence == "ns":
                                ps_data = ps
                                ps_reward = self.reward_function(
                                    ps_data
                                ).detach()  # size = [batch_size,1]
                            elif self.args.reward_dependence == "sans":
                                ps_data = torch.cat([states, actions, ps], dim=-1)
                                ps_reward = self.reward_function(
                                    ps_data
                                ).detach()  # size = [batch_size,1]
                        else:
                            ps_reward = rewards

                        all_predict_Q[:, ee * self.args.p_sample_n + ppp] = (
                            self.target_critic(
                                ps, self.target_actor(ps).detach()
                            ).squeeze(-1)
                        )
                        backup = (
                            ps_reward
                            + (1 - dones)
                            * self.discount
                            * self.target_critic(ps, self.target_actor(ps).detach())
                        ).squeeze(-1)
                        # size = [ batch_size]
                        all_predict_backup[:, ee * self.args.p_sample_n + ppp] = (
                            copy.deepcopy(backup.detach())
                        )

            elif self.args.sample_from_avg:
                for ee in range(self.n_ensemble):
                    pred_ls = self.forward_model_l[ee](states, actions)
                    pred_us = self.forward_model_u[ee](states, actions)
                    # For record
                    all_predict_US[:, :, ee] = copy.deepcopy(pred_us.detach())
                    all_predict_LS[:, :, ee] = copy.deepcopy(pred_ls.detach())
                avg_US = torch.mean(all_predict_US, dim=-1)
                avg_LS = torch.mean(all_predict_LS, dim=-1)
                for ee in range(self.n_ensemble):
                    for ppp in range(self.args.p_sample_n):
                        ps = self.sample_ns(avg_LS, avg_US)
                        all_predict_states[:, :, ee * self.args.p_sample_n + ppp] = (
                            copy.deepcopy(ps.detach())
                        )
                        if self.args.learn_reward:
                            if self.args.reward_dependence == "ns":
                                ps_data = ps
                                ps_reward = self.reward_function(
                                    ps_data
                                ).detach()  # size = [batch_size,1]
                            elif self.args.reward_dependence == "sans":
                                ps_data = torch.cat([states, actions, ps], dim=-1)
                                ps_reward = self.reward_function(
                                    ps_data
                                ).detach()  # size = [batch_size,1]
                        else:
                            ps_reward = rewards

                        all_predict_Q[:, ee * self.args.p_sample_n + ppp] = (
                            self.target_critic(
                                ps, self.target_actor(ps).detach()
                            ).squeeze(-1)
                        )
                        backup = (
                            ps_reward
                            + (1 - dones)
                            * self.discount
                            * self.target_critic(ps, self.target_actor(ps).detach())
                        ).squeeze(-1)
                        # size = [ batch_size]
                        all_predict_backup[:, ee * self.args.p_sample_n + ppp] = (
                            copy.deepcopy(backup.detach())
                        )

            elif self.args.sample_from_bound:
                for ee in range(self.n_ensemble):
                    pred_ls = self.forward_model_l[ee](states, actions)
                    pred_us = self.forward_model_u[ee](states, actions)
                    # For record
                    all_predict_US[:, :, ee] = copy.deepcopy(pred_us.detach())
                    all_predict_LS[:, :, ee] = copy.deepcopy(pred_ls.detach())
                B_US = torch.max(all_predict_US, dim=-1)[
                    0
                ]  # shape = [batch_size, state_space]
                B_LS = torch.min(all_predict_LS, dim=-1)[0]

                for ee in range(self.n_ensemble):
                    for ppp in range(self.args.p_sample_n):
                        ps = self.sample_ns(B_LS, B_US)
                        all_predict_states[:, :, ee * self.args.p_sample_n + ppp] = (
                            copy.deepcopy(ps.detach())
                        )
                        if self.args.learn_reward:
                            if self.args.reward_dependence == "ns":
                                ps_data = ps
                                ps_reward = self.reward_function(
                                    ps_data
                                ).detach()  # size = [batch_size,1]
                            elif self.args.reward_dependence == "sans":
                                ps_data = torch.cat([states, actions, ps], dim=-1)
                                ps_reward = self.reward_function(
                                    ps_data
                                ).detach()  # size = [batch_size,1]
                        else:
                            ps_reward = rewards

                        all_predict_Q[:, ee * self.args.p_sample_n + ppp] = (
                            self.target_critic(
                                ps, self.target_actor(ps).detach()
                            ).squeeze(-1)
                        )
                        backup = (
                            ps_reward
                            + (1 - dones)
                            * self.discount
                            * self.target_critic(ps, self.target_actor(ps).detach())
                        ).squeeze(-1)
                        # size = [ batch_size]
                        all_predict_backup[:, ee * self.args.p_sample_n + ppp] = (
                            copy.deepcopy(backup.detach())
                        )

            next_Q = (
                self.target_critic(next_states, self.target_actor(next_states))
                .detach()
                .squeeze(-1)
            )
            all_predict_Q[:, self.args.n_ensemble * self.args.p_sample_n] = (
                copy.deepcopy(next_Q)
            )
            all_predict_backup[:, self.args.n_ensemble * self.args.p_sample_n] = (
                rewards
                + (1 - dones) * self.discount * copy.deepcopy(next_Q.unsqueeze(-1))
            ).squeeze(-1)
            all_predict_states[:, :, self.args.n_ensemble * self.args.p_sample_n] = (
                copy.deepcopy(next_states.detach().cpu())
            )

            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< back-up value >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            max_index = np.argmax(all_predict_backup, axis=-1)  # size = batch_size
            max_state = all_predict_states[
                np.arange(batch_size), :, max_index
            ]  # size = [ batch_size, state_space ]
            max_backup = (
                torch.FloatTensor(all_predict_backup[np.arange(batch_size), max_index])
                .unsqueeze(-1)
                .to(self.device)
            )  # size = [batch_size, 1]

            for ccc in range(self.args.IterOnlyCritic_n):
                critic_loss = F.mse_loss(self.critic(states, actions), max_backup)
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

            ###### checked whether quantile model capture the all the next state #####
            predict_US = torch.max(all_predict_US, dim=-1)[
                0
            ]  # shape = [batch_size, state_space]
            predict_LS = torch.min(all_predict_LS, dim=-1)[0]
            precent_in, mean_jud = self.declare_in(predict_US, predict_LS, next_states)

            large_dis, small_dis = self.compute_dis_bound(
                predict_US, predict_LS, next_states
            )

            if self.args.sample_from_avg:
                precent_in_avg, mean_jud_avg = self.declare_in(
                    avg_US, avg_LS, next_states
                )
                large_dis_avg, small_dis_avg = self.compute_dis_bound(
                    avg_US, avg_LS, next_states
                )

            if self.args.use_wandb and id == 0:
                wandb.log(
                    {
                        f"Agent_{id}_Precentage of next hidden state in quantile predicted bound": precent_in
                    },
                    step=total_steps,
                )
                wandb.log(
                    {
                        f"Agent_{id}_Averaged Q0-value over this batch": self.critic(
                            states, self.actor(states)
                        )
                        .mean()
                        .detach()
                    },
                    step=total_steps,
                )
                wandb.log(
                    {
                        f"Agent_{id}_mean dimension in quantile predicted bound": mean_jud
                    },
                    step=total_steps,
                )
                wandb.log(
                    {
                        f"Agent_{id}_mean distance between larger ns and upper bound": large_dis
                    },
                    step=total_steps,
                )
                wandb.log(
                    {
                        f"Agent_{id}_mean distance between smaller ns and lower bound": small_dis
                    },
                    step=total_steps,
                )

                if self.args.sample_from_avg:
                    wandb.log(
                        {
                            f"Agent_{id}_Precentage of next state in AVERAGE quantile predicted bound": precent_in_avg
                        },
                        step=total_steps,
                    )
                    wandb.log(
                        {
                            f"Agent_{id}_mean dimension in AVERAGE quantile predicted bound": mean_jud_avg
                        },
                        step=total_steps,
                    )
                    wandb.log(
                        {
                            f"Agent_{id}_mean distance between larger ns and AVERAGE upper bound": large_dis_avg
                        },
                        step=total_steps,
                    )
                    wandb.log(
                        {
                            f"Agent_{id}_mean distance between smaller ns and AVERAGE lower bound": small_dis_avg
                        },
                        step=total_steps,
                    )

            # <<<<< Target network
            for param, target_param in zip(
                self.critic.parameters(), self.target_critic.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

            for param, target_param in zip(
                self.actor.parameters(), self.target_actor.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

        return

    # Save some experiences from the replay buffer to do further analysis
    def save_buffer(self, buffer, size=1000):
        states, actions, rewards, next_states, dones = buffer.getBatch(size)
        experience_dict = {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "next_states": next_states,
            "dones": dones,
        }
        return experience_dict

    def declare_in(
        self, pred_quan_u, pred_quan_l, next_states
    ):  # next_states shape = [batch_size, state_space]
        ss = pred_quan_u.shape[0]
        total_size = torch.tensor(ss, dtype=torch.float32)
        # print(total_size)
        pred_quan_u = torch.FloatTensor(pred_quan_u).to(self.device)
        pred_quan_l = torch.FloatTensor(pred_quan_l).to(self.device)
        jud = (next_states <= pred_quan_u) & (
            next_states >= pred_quan_l
        )  # if == 1, then is in;  [batch_size, state_space ]
        jud = torch.sum(jud, dim=-1)
        jud_all = (
            jud >= self.args.state_space
        )  # if in, 1; otherwise, 0 shape [batch_size]
        precent_in = 100 * (torch.sum(jud_all) / total_size)
        mean_jud = torch.mean(jud.float())

        return precent_in, mean_jud

    def compute_dis_bound(self, pred_quan_u, pred_quan_l, next_states):
        pred_quan_u = torch.FloatTensor(pred_quan_u).to(self.device)
        pred_quan_l = torch.FloatTensor(pred_quan_l).to(self.device)
        jud_large = (
            next_states > pred_quan_u
        )  # [batch_size, state_space] record each dimension and compare with pred_quan_u
        jud_small = next_states < pred_quan_l

        # print("large", torch.nonzero(jud_large))
        # print("small", torch.nonzero(jud_small))

        large_dis = torch.mean((next_states[jud_large] - pred_quan_u[jud_large]))
        small_dis = torch.mean((pred_quan_l[jud_small] - next_states[jud_small]))

        return large_dis, small_dis

    def save(self, filename, agent_idx):
        torch.save(
            self.critic.state_dict(), filename + "_" + str(agent_idx) + "_critic"
        )
        torch.save(
            self.target_critic.state_dict(),
            filename + "_" + str(agent_idx) + "_target_critic",
        )
        torch.save(self.actor.state_dict(), filename + "_" + str(agent_idx) + "_actor")
        torch.save(
            self.target_actor.state_dict(),
            filename + "_" + str(agent_idx) + "_target_actor",
        )
        for j in range(self.args.n_ensemble):
            torch.save(
                self.forward_model_l[j].state_dict(),
                filename + "_" + str(agent_idx) + f"_forward_model_l{j}",
            )
            torch.save(
                self.forward_model_u[j].state_dict(),
                filename + "_" + str(agent_idx) + f"_forward_model_u{j}",
            )
        if self.args.learn_reward:
            torch.save(
                self.reward_function.state_dict(),
                filename + "_" + str(agent_idx) + "_RewardFunction",
            )

    def load(self, filename, agent_idx):
        self.critic.load_state_dict(
            torch.load(filename + "_" + str(agent_idx) + "_critic")
        )
        self.target_critic.load_state_dict(
            torch.load(filename + "_" + str(agent_idx) + "_target_critic")
        )
        self.actor.load_state_dict(
            torch.load(filename + "_" + str(agent_idx) + "_actor")
        )
        self.target_actor.load_state_dict(
            torch.load(filename + "_" + str(agent_idx) + "_target_actor")
        )
        for j in range(self.args.n_ensemble):
            self.forward_model_l[j].load_state_dict(
                torch.load(filename + "_" + str(agent_idx) + f"_forward_model_l{j}")
            )
            self.forward_model_u[j].load_state_dict(
                torch.load(filename + "_" + str(agent_idx) + f"_forward_model_u{j}")
            )
        if self.args.learn_reward:
            self.reward_function.load_state_dict(
                torch.load(filename + "_" + str(agent_idx) + "_RewardFunction")
            )
