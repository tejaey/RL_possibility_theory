import logging
import math
import random
from abc import ABC, abstractmethod
from typing import Literal

import gymnasium
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import FloatType, optim

from config import DEVICE
from qnets import (EnsembleQuantileModels, MeanStdPredictor, RewardModel,
                   StdPredictor)


def unpack_batch(batch):
    states, actions, rewards, next_states, dones = batch

    states_t = torch.FloatTensor(states).to(DEVICE)
    actions_t = torch.LongTensor(actions).unsqueeze(-1).to(DEVICE)
    rewards_t = torch.FloatTensor(rewards).unsqueeze(-1).to(DEVICE)
    next_states_t = torch.FloatTensor(next_states).to(DEVICE)
    dones_t = torch.FloatTensor(dones).unsqueeze(-1).to(DEVICE)

    return states_t, actions_t, rewards_t, next_states_t, dones_t


class td_loss_meta(ABC):
    def __init__(self):
        self.lossfunc = nn.MSELoss()

    @abstractmethod
    def __call__(self, batch, online_qnet, target_qnet, optimizers) -> float:
        pass


class distributional_qn_loss(td_loss_meta):
    def __init__(self, method: Literal["Dkl", "Wasserstein"], GAMMA):
        super().__init__()  # call parent first
        self.method = method
        self.GAMMA = GAMMA

    def __call__(self, batch, online_qnet, target_qnet, optimizers) -> float:
        states_t, actions_t, rewards_t, next_states_t, dones_t = unpack_batch(batch)

        mean, logvar = online_qnet(states_t)
        mean_taken = mean.gather(1, actions_t)
        logvar_taken = logvar.gather(1, actions_t)
        var_taken = torch.exp(logvar_taken)

        with torch.no_grad():
            next_mean, _ = target_qnet(next_states_t)
            next_max = next_mean.max(dim=1, keepdim=True)[0]
            td_target = rewards_t + self.GAMMA * next_max * (1 - dones_t)

        match self.method:
            case "Dkl":
                diff = mean_taken - td_target
                loss = (diff.pow(2) + var_taken).mean()
            case "Wasserstein":
                diff_sq = (td_target - mean_taken).pow(2)
                kl_term = 0.5 * (logvar_taken + np.log(2 * np.pi)) + diff_sq / (
                    2 * var_taken
                )
                loss = kl_term.mean()
            case _:
                raise ValueError(f"Unknown method {self.method!r}")

        optimizers.zero_grad()
        loss.backward()
        optimizers.step()
        return loss.item()


class td_loss_ensemble(td_loss_meta):
    def __init__(
        self,
        GAMMA,
        ALPHA,
        possibility_update: Literal[
            "mle", "no_update", "avg_likelihood_ema", "mle_max_update"
        ],
        use_ensemble_min: bool = False,
        normalise: bool = False,
    ):
        super().__init__()
        self.ALPHA = ALPHA
        self.GAMMA = GAMMA
        self.possibility_update = possibility_update
        self.normalise = normalise
        self.use_ensemble_min = use_ensemble_min

    def __call__(self, batch, online_qnet, target_qnet, optimizers) -> float:
        states_t, actions_t, rewards_t, next_states_t, dones_t = unpack_batch(batch)
        losses = []
        likelihoods = []

        # next_q_values should be overwritten
        next_q_values = None
        if self.use_ensemble_min:
            with torch.no_grad():
                next_q_values = torch.amin(
                    torch.stack(
                        [
                            target_qnet.qnets[i](next_states_t).max(
                                dim=1, keepdim=True
                            )[0]
                            for i in range(target_qnet.num_ensemble)
                        ]
                    ),
                    dim=0,
                )
        for i, qnet in enumerate(online_qnet.qnets):
            optimizer = optimizers[i]

            # Current Q-values for chosen actions
            current_q_values = qnet(states_t).gather(1, actions_t)
            if not self.use_ensemble_min:
                with torch.no_grad():
                    next_q_values = target_qnet.qnets[i](next_states_t).max(
                        dim=1, keepdim=True
                    )[0]
            expected_q_values = rewards_t + self.GAMMA * next_q_values * (1 - dones_t)
            loss = self.lossfunc(current_q_values, expected_q_values)
            losses.append(loss.item())
            likelihood = math.exp(-loss.item())
            likelihoods.append(likelihood)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        match self.possibility_update:
            case "mle":
                max_likelihood = (
                    max(
                        [
                            likelihoods[i] * online_qnet.possibility[i]
                            for i in range(len(likelihoods))
                        ]
                    )
                    + 1e-8
                )
                for i in range(online_qnet.num_ensemble):
                    online_qnet.possibility[i] = (
                        online_qnet.possibility[i] * likelihoods[i] / max_likelihood
                    )

            case "mle_max_update":
                max_likelihood = (
                    max(
                        [
                            likelihoods[i] * online_qnet.possibility[i]
                            for i in range(len(likelihoods))
                        ]
                    )
                    + 1e-8
                )
                for i in range(online_qnet.num_ensemble):
                    # we should not need this min here
                    online_qnet.possibility[i] = min(
                        max(
                            0.9 * online_qnet.possibility[i],
                            likelihoods[i] / max_likelihood,
                        ),
                        1,
                    )

            case "avg_likelihood_ema":
                print("this should not be in use")
                # model weights remain close to 1.
                avg_likelihood = sum(likelihoods) / len(likelihoods) + 1e-8
                for i in range(online_qnet.num_ensemble):
                    candidate = likelihoods[i] / avg_likelihood
                    online_qnet.possibility[i] = max(
                        min(
                            (1 - self.ALPHA) * online_qnet.possibility[i]
                            + self.ALPHA * candidate,
                            1,
                        ),
                        0.1,
                    )
            case "no_update":
                online_qnet.possibility = [1] * len(online_qnet.possibility)
            case _:
                raise Exception("Invalid possibility update")

        max_possibility = max(online_qnet.possibility)
        online_qnet.possibility = [
            (i / max_possibility) for i in online_qnet.possibility
        ]
        online_qnet.possibility = [max(i, 0.001) for i in online_qnet.possibility]

        if self.normalise:
            s = sum(online_qnet.possibility)
            if s == 0:
                logging.error(online_qnet.possibility)
                s += 1e-16
            online_qnet.possibility = [x / s for x in online_qnet.possibility]
        return float(np.mean(losses))


class td_loss_single_dqn(td_loss_meta):
    def __init__(self, GAMMA):
        self.GAMMA = GAMMA
        super().__init__()
        pass

    def __call__(self, batch, online_qnet, target_qnet, optimizers) -> float:
        states_t, actions_t, rewards_t, next_states_t, dones_t = unpack_batch(batch)
        optimizer = optimizers[0]
        current_q_values = online_qnet(states_t).gather(1, actions_t)
        with torch.no_grad():
            next_q_values = target_qnet(next_states_t).max(dim=1, keepdim=True)[0]

        expected_q_values = rewards_t + self.GAMMA * next_q_values * (1 - dones_t)
        loss = self.lossfunc(current_q_values, expected_q_values)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()


def quantile_loss(pred, target, tau):
    error = target - pred
    loss = torch.max((tau - 1) * error, tau * error)
    return loss.mean()


def sample_candidates_triangular(lower, upper, num_samples):
    """
    Sample candidate next states from the interval [lower, upper] using a symmetric triangular distribution.
    """
    batch_size, state_dim = lower.shape
    u = torch.rand(batch_size, num_samples, state_dim, device=lower.device)

    lower_expanded = lower.unsqueeze(1)
    upper_expanded = upper.unsqueeze(1)
    diff = upper_expanded - lower_expanded
    samples = torch.where(
        u < 0.5,
        lower_expanded + diff * torch.sqrt(u / 2.0),
        upper_expanded - diff * torch.sqrt((1 - u) / 2.0),
    )
    return samples


def sample_candidates_unif(lower, upper, num_samples):
    """
    Uniformly sample candidate next states from the interval [lower, upper].
    - lower, upper: Tensors of shape [batch_size, state_dim]
    - Returns: Tensor of shape [batch_size, num_samples, state_dim]
    """
    batch_size, state_dim = lower.shape
    uniform_samples = torch.rand(batch_size, num_samples, state_dim)
    candidates = lower.unsqueeze(1) + (upper - lower).unsqueeze(1) * uniform_samples
    return candidates


def gaussian_std_nll_loss_bak(predicted_log_std, error):
    std = torch.exp(predicted_log_std)
    const = torch.tensor(2 * torch.pi, device=predicted_log_std.device)
    loss = 0.5 * torch.log(const) + predicted_log_std + 0.5 * ((error) / std) ** 2
    return loss.mean()


def gaussian_std_nll_loss(
    mu: torch.Tensor, std: torch.Tensor, target: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    var = std.pow(2) + eps
    nll = 0.5 * (torch.log(2 * torch.pi * var) + (target - mu).pow(2) / var)
    return nll.mean()


def sample_candidates_gaussian(std_model, states, actions, next_states, num_samples):
    """
    Sample candidate next states using a Gaussian model that predicts the std.
    The mean is taken as the observed next state.

    Args:
        std_model: an instance of StdPredictor.
        states: Tensor of shape [batch, state_dim]
        actions: Tensor of shape [batch, action_dim]
        next_states: Tensor of shape [batch, state_dim] (observed next states)
        num_samples: number of samples per instance.
    Returns:
        Tensor of shape [batch, num_samples, state_dim].
    """
    log_std = std_model(states, actions)
    std = torch.exp(log_std)

    batch_size, state_dim = next_states.shape
    eps = torch.randn(batch_size, num_samples, state_dim, device=next_states.device)
    mean_expanded = next_states.unsqueeze(1)
    std_expanded = std.unsqueeze(1)

    samples = mean_expanded + std_expanded * eps
    return samples


class actor_critic_loss:
    def __init__(self, batch_size, gamma=0.9):
        self.batch_size = batch_size
        self.gamma = gamma
        # I dont think we ever use this?
        # self.state_dim = state_dim
        # self.action_dim = action_dim

    def __call__(
        self,
        replay_buffer,
        online_actor,
        target_actor,
        actor_optimizer,
        online_critic,
        target_critic,
        critic_optimizer,
        **kwargs,
    ):
        gamma = self.gamma
        batch_size = self.batch_size

        if len(replay_buffer) < batch_size:
            return -1

        states_np, actions_np, rewards_np, next_states_np, dones_np = (
            replay_buffer.sample(batch_size)
        )
        states = torch.FloatTensor(states_np).to(DEVICE)
        actions = torch.FloatTensor(actions_np).to(DEVICE)
        rewards = torch.FloatTensor(rewards_np).unsqueeze(1).to(DEVICE)
        next_states = torch.FloatTensor(next_states_np).to(DEVICE)
        dones = torch.FloatTensor(dones_np).unsqueeze(1).to(DEVICE)

        next_actions = target_actor(next_states)
        next_q = target_critic(next_states, next_actions)
        target_q = rewards + gamma * (1 - dones) * next_q.detach()

        current_q = online_critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        actor_loss = -online_critic(states, online_actor(states)).mean()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        return actor_loss


class ActorCriticLossMaxMaxFix_zerostep:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        batch_size: int,
        gamma: float = 0.9,
        num_neighbour_sample: int = 5,
        num_next_state_sample: int = 3,
        next_state_sample: Literal["quantile", "gaussian", "null"] = "quantile",
        rl_p: float = 0.1,
        epsilon: float = 0.01,
        use_min: bool = False,
        **kwargs,
    ):
        """
        Zero‐step possibilistic actor‐critic loss:
        - With probability (1 - rollout_prob) perform standard TD(0).
        - Otherwise perform possibilistic model‐based planning via neighbour + next‐state sampling.
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.num_neighbour_sample = num_neighbour_sample
        self.num_next_state_sample = num_next_state_sample
        self.next_state_sample = next_state_sample
        self.rl_p = 1 - rl_p
        self.epsilon = epsilon
        self.use_min = use_min

        self.reward_model = RewardModel(state_dim).to(DEVICE)
        self.reward_opt = optim.Adam(self.reward_model.parameters(), lr=1e-3)

        if next_state_sample == "quantile":
            self.quantile_models = EnsembleQuantileModels(
                state_dim, action_dim, quantiles=[0.05, 0.95], hidden_dim=128
            ).to(DEVICE)
            self.quantile_opts = [
                optim.Adam(m.parameters(), lr=1e-3) for m in self.quantile_models.models
            ]
        elif next_state_sample == "gaussian":
            self.next_state_model = MeanStdPredictor(
                state_dim, action_dim, hidden_dim=128
            ).to(DEVICE)
            self.next_state_opt = optim.Adam(
                self.next_state_model.parameters(), lr=1e-3
            )

    def compute_yj(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
        target_actor: nn.Module,
        target_critic: nn.Module,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        bs = states.size(0)
        γ = self.gamma

        with torch.no_grad():
            a1 = target_actor(next_states)
            q1 = target_critic(next_states, a1)
        y0 = (rewards + γ * (1 - dones) * q1).unsqueeze(1)

        if self.next_state_sample == "quantile":
            for τ, m, opt in zip(
                self.quantile_models.quantiles,
                self.quantile_models.models,
                self.quantile_opts,
            ):
                q_hat = m(states, actions)
                loss_q = quantile_loss(q_hat, next_states, tau=τ)
                opt.zero_grad()
                loss_q.backward()
                opt.step()
        elif self.next_state_sample == "gaussian":
            mu, std = self.next_state_model(states, actions)
            loss_g = gaussian_std_nll_loss(mu, std, next_states)
            self.next_state_opt.zero_grad()
            loss_g.backward()
            self.next_state_opt.step()

        if random.random() < self.rl_p:
            return y0

        y_plan = []
        for _ in range(self.num_neighbour_sample):
            eps = (
                torch.rand(bs, self.state_dim, device=states.device) * 2 - 1
            ) * self.epsilon
            s_star = states + eps
            a_star = target_actor(s_star)

            y_k_vals = []
            for _ in range(self.num_next_state_sample):
                if self.next_state_sample == "quantile":
                    ql, qu = self.quantile_models(s_star, a_star)
                    u = torch.rand_like(ql)
                    s_prime = (1 - u) * ql + u * qu
                else:
                    mu, std = self.next_state_model(s_star, a_star)
                    s_prime = mu + std * torch.randn_like(std)

                r_pred = self.reward_model(s_star, s_prime)
                a2 = target_actor(s_prime)
                q2 = target_critic(s_prime, a2)

                y_i = r_pred + γ * q2
                y_k_vals.append(y_i)

            y_k = torch.stack(y_k_vals, dim=1).max(dim=1, keepdim=True)[0]
            y_plan.append(y_k.unsqueeze(2))

        Y_plan = torch.cat(y_plan, dim=1)
        return Y_plan

    def __call__(
        self,
        replay_buffer,
        online_actor: nn.Module,
        target_actor: nn.Module,
        actor_optimizer: optim.Optimizer,
        online_critic: nn.Module,
        target_critic: nn.Module,
        critic_optimizer: optim.Optimizer,
        **kwargs,
    ):
        if len(replay_buffer) < self.batch_size:
            return None

        s, a, r, s_next, done = replay_buffer.sample(self.batch_size)
        states = torch.FloatTensor(s).to(DEVICE)
        actions = torch.FloatTensor(a).to(DEVICE)
        rewards = torch.FloatTensor(r).unsqueeze(-1).to(DEVICE)
        next_states = torch.FloatTensor(s_next).to(DEVICE)
        dones = torch.FloatTensor(done).unsqueeze(-1).to(DEVICE)

        r_pred = self.reward_model(states, next_states)
        loss_r = F.mse_loss(r_pred, rewards)
        self.reward_opt.zero_grad()
        loss_r.backward()
        self.reward_opt.step()

        Y = self.compute_yj(
            states, actions, next_states, target_actor, target_critic, rewards, dones
        )
        Y_agg, _ = torch.max(Y, dim=1)
        target_q = Y_agg.detach()

        current_q = online_critic(states, actions)
        loss_c = F.mse_loss(current_q, target_q)
        critic_optimizer.zero_grad()
        loss_c.backward()
        critic_optimizer.step()

        loss_a = -online_critic(states, online_actor(states)).mean()
        actor_optimizer.zero_grad()
        loss_a.backward()
        actor_optimizer.step()

        return loss_a


class ActorCriticLossMaxMaxFix_onestep:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        batch_size: int,
        gamma: float = 0.9,
        num_next_state_sample: int = 10,
        next_state_sample: Literal["quantile", "gaussian", "null"] = "quantile",
        rollout_depth: int = 0,  # only 0 or 1 supported
        use_min: bool = False,
        rl_p: float = 0.1,
        **kwargs,
    ):
        assert rollout_depth in (0, 1), "Only rollout_depth=0 or 1 supported"
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.next_state_sample = next_state_sample
        self.rollout_depth = rollout_depth
        self.use_min = use_min
        self.num_next_state_sample = num_next_state_sample
        self.rl_p = 1 - rl_p
        self.reward_model = RewardModel(state_dim).to(DEVICE)
        self.reward_opt = optim.Adam(self.reward_model.parameters(), lr=1e-3)

        if rollout_depth == 1 and next_state_sample == "quantile":
            self.quantile_models = EnsembleQuantileModels(
                state_dim, action_dim, quantiles=[0.05, 0.95], hidden_dim=128
            ).to(DEVICE)
            self.quantile_opts = [
                optim.Adam(m.parameters(), lr=1e-3) for m in self.quantile_models.models
            ]
        elif rollout_depth == 1 and next_state_sample == "gaussian":
            self.next_state_model = MeanStdPredictor(
                state_dim, action_dim, hidden_dim=128
            ).to(DEVICE)
            self.next_state_opt = optim.Adam(
                self.next_state_model.parameters(), lr=1e-3
            )

    def compute_yj(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
        target_actor: nn.Module,
        target_critic: nn.Module,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        bs = states.size(0)
        γ = self.gamma

        with torch.no_grad():
            a1 = target_actor(next_states)
            q1 = target_critic(next_states, a1)
        y0 = (rewards + γ * (1 - dones) * q1).unsqueeze(1)

        if self.rollout_depth == 0 or self.next_state_sample == "null":
            return y0  # (bs,1,1)

        # Train the imagination model on real data:
        if self.next_state_sample == "quantile":
            for τ, m, opt in zip(
                self.quantile_models.quantiles,
                self.quantile_models.models,
                self.quantile_opts,
            ):
                q_hat = m(states, actions)
                loss_q = quantile_loss(q_hat, next_states, tau=τ)
                opt.zero_grad()
                loss_q.backward()
                opt.step()
        else:  # gaussian
            mu, std = self.next_state_model(states, actions)
            loss_g = gaussian_std_nll_loss(mu, std, next_states)
            self.next_state_opt.zero_grad()
            loss_g.backward()
            self.next_state_opt.step()

        if random.random() < self.rl_p:
            return y0
        with torch.no_grad():
            N = self.num_next_state_sample
            s1 = next_states.unsqueeze(1).expand(bs, N, self.state_dim)  # (bs,N,D)
            a1_rep = target_actor(s1.reshape(-1, self.state_dim))  # (bs*N, A)

            if self.next_state_sample == "quantile":
                ql, qu = self.quantile_models(s1.reshape(-1, self.state_dim), a1_rep)
                eps = torch.rand_like(ql)
                s2_flat = (1 - eps) * ql + eps * qu
            else:
                mu, std = self.next_state_model(s1.reshape(-1, self.state_dim), a1_rep)
                s2_flat = mu + std * torch.randn_like(std)

            s2 = s2_flat.view(bs, N, self.state_dim)

            r1 = self.reward_model(s1.reshape(-1, self.state_dim), s2_flat).view(
                bs, N, 1
            )

            a2 = target_actor(s2_flat)
            q2 = target_critic(s2_flat, a2).view(bs, N, 1)

            imag_term = r1 + γ * q2

        y1 = rewards.unsqueeze(1) + γ * imag_term

        return y1

    def __call__(
        self,
        replay_buffer,
        online_actor,
        target_actor,
        actor_optimizer,
        online_critic,
        target_critic,
        critic_optimizer,
        **kwargs,
    ):
        if len(replay_buffer) < self.batch_size:
            return None

        s, a, r, s_next, done = replay_buffer.sample(self.batch_size)
        states = torch.FloatTensor(s).to(DEVICE)
        actions = torch.FloatTensor(a).to(DEVICE)
        rewards = torch.FloatTensor(r).unsqueeze(-1).to(DEVICE)
        next_states = torch.FloatTensor(s_next).to(DEVICE)
        dones = torch.FloatTensor(done).unsqueeze(-1).to(DEVICE)

        r_pred = self.reward_model(states, next_states)
        loss_r = F.mse_loss(r_pred, rewards)
        self.reward_opt.zero_grad()
        loss_r.backward()
        self.reward_opt.step()

        Y = self.compute_yj(
            states, actions, next_states, target_actor, target_critic, rewards, dones
        )

        if self.use_min:
            Y_agg, _ = torch.min(Y, dim=1)
        else:
            Y_agg, _ = torch.max(Y, dim=1)

        target_q = Y_agg.detach()
        current_q = online_critic(states, actions)
        loss_c = F.mse_loss(current_q, target_q)
        critic_optimizer.zero_grad()
        loss_c.backward()
        critic_optimizer.step()

        loss_a = -online_critic(states, online_actor(states)).mean()
        actor_optimizer.zero_grad()
        loss_a.backward()
        actor_optimizer.step()

        return loss_a
