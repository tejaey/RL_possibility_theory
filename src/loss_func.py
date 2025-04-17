from typing import Literal
import gymnasium
import torch.nn.functional as F
import torch, random, math
from torch import FloatType, optim
import numpy as np
import torch.nn as nn
from abc import ABC, abstractmethod
import logging

from qnets import StdPredictor, RewardModel
from config import DEVICE


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

        # 4) Compute loss by method
        match self.method:
            case "Dkl":
                # DKL variant uses (μ - target)^2 + σ^2
                diff = mean_taken - td_target
                loss = (diff.pow(2) + var_taken).mean()
            case "Wasserstein":
                # Wasserstein variant uses KL-style term
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


class td_loss_ensemble_grad(td_loss_meta):
    """
    Attributes:
        ALPHA (float): EMA parameter (good range: 0.1 - 0.3)
        GAMMA (float): reward discount in the environment
        BETA (float): Weight of the grad_norm in 'exp(-loss.item() - BETA * grad_norm)' (good range: 0.01)
    """

    def __init__(self, GAMMA, ALPHA, BETA, normalise):
        super().__init__()
        self.ALPHA = ALPHA
        self.BETA = BETA
        self.GAMMA = GAMMA
        self.normalise = normalise

    def __call__(self, batch, online_qnet, target_qnet, optimizers) -> float:
        states_t, actions_t, rewards_t, next_states_t, dones_t = unpack_batch(batch)
        losses = []
        candidate_poss = []
        for i, qnet in enumerate(online_qnet.qnets):
            optimizer = optimizers[i]

            # Compute current Q-values for the taken actions.
            current_q_values = qnet(states_t).gather(1, actions_t)

            # Each ensemble member has its own target network.

            with torch.no_grad():
                next_q_values = target_qnet.qnets[i](next_states_t).max(
                    dim=1, keepdim=True
                )[0]

            expected_q_values = rewards_t + self.GAMMA * next_q_values * (1 - dones_t)
            loss = self.lossfunc(current_q_values, expected_q_values)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()

            grad_norm = 0.0
            for param in qnet.parameters():
                if param.grad is not None:
                    grad_norm += param.grad.data.norm(2).item() ** 2
            grad_norm = math.sqrt(grad_norm)

            candidate = math.exp(-loss.item() - self.BETA * grad_norm)
            candidate_poss.append(candidate)

            optimizer.step()

        # Update possibility for each ensemble member using an EMA update.
        for i in range(online_qnet.num_ensemble):
            online_qnet.possibility[i] = (1 - self.ALPHA) * online_qnet.possibility[
                i
            ] + self.ALPHA * candidate_poss[i]

        online_qnet.possibility = [max(i, 0.001) for i in online_qnet.possibility]
        if self.normalise:
            s = sum(online_qnet.possibility)
            online_qnet.possibility = [x / s for x in online_qnet.possibility]
        return float(np.mean(losses))


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
                            0.99 * online_qnet.possibility[i],
                            likelihoods[i] / max_likelihood,
                        ),
                        1,
                    )

            case "avg_likelihood_ema":
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
            # case "avg_likelihood_ema2":
            #     # model weights remain close to 1.
            #     avg_likelihood = sum(likelihoods) / len(likelihoods) + 1e-8
            #     for i in range(online_qnet.num_ensemble):
            #         candidate = likelihoods[i] / avg_likelihood
            #         online_qnet.possibility[i] = min(
            #             online_qnet.possibility[i] * candidate, 1
            #         )
            case "no_update":
                online_qnet.possibility = [1] * len(online_qnet.possibility)
            case _:
                raise Exception("Invalid possibility update")

        # alpha = 0.1  # Smoothing factor (0 < alpha < 1)
        # avg_likelihood = sum(likelihoods) / len(likelihoods) + 0.000001
        # for i in range(online_qnet.num_ensemble):
        #     candidate = likelihoods[i] / avg_likelihood  # >1 if better than average, <1 if worse
        #     # Update with EMA: this increases possibility if candidate > 1, and decreases if < 1
        #     online_qnet.possibility[i] = max(min(online_qnet.possibility[i]* candidate, 1), 0.1)
        # total = sum(online_qnet.possibility)
        # online_qnet.possibility = [p / total for p in online_qnet.possibility]
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


def gaussian_std_nll_loss(predicted_log_std, error):
    std = torch.exp(predicted_log_std)
    const = torch.tensor(2 * torch.pi, device=predicted_log_std.device)
    loss = 0.5 * torch.log(const) + predicted_log_std + 0.5 * ((error) / std) ** 2
    return loss.mean()


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
    # Predict log_std for each (state, action)
    log_std = std_model(states, actions)  # [batch, state_dim]
    std = torch.exp(log_std)  # [batch, state_dim]

    batch_size, state_dim = next_states.shape
    # Sample epsilon from standard normal: [batch, num_samples, state_dim]
    eps = torch.randn(batch_size, num_samples, state_dim, device=next_states.device)
    # Expand mean (observed next_states) and std to sample shape:
    mean_expanded = next_states.unsqueeze(1)  # [batch, 1, state_dim]
    std_expanded = std.unsqueeze(1)  # [batch, 1, state_dim]

    samples = mean_expanded + std_expanded * eps
    return samples


class ensemble_critic_loss:
    def __init__(self, state_dim, action_dim, batch_size, gamma=0.9):
        self.batch_size = batch_size
        self.gamma = gamma
        self.state_dim = state_dim
        self.action_dim = action_dim

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

        # Compute the target Q-value using the target networks.
        next_actions = target_actor(next_states)
        next_q = target_critic(next_states, next_actions)
        target_q = rewards + gamma * (1 - dones) * next_q.detach()

        # Critic update: minimize the MSE between current Q and target Q.
        current_q = online_critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        # Actor update: maximize the Q-value by minimizing the negative Q.
        actor_loss = -online_critic(states, online_actor(states)).mean()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        return actor_loss


class actor_critic_loss_maxmax:
    def __init__(
        self,
        state_dim,
        action_dim,
        batch_size,
        gamma=0.9,
        num_next_state_sample=10,
        next_state_sample: Literal["gaussian", "triangular", "unif", "null"] = "unif",
        use_min: bool = False,
    ):
        self.batch_size = batch_size
        self.gamma = gamma
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_next_state_sample = num_next_state_sample
        self.next_state_sample = next_state_sample
        self.std_model = StdPredictor(state_dim=state_dim, action_dim=action_dim).to(
            DEVICE
        )
        self.std_optimizer = optim.Adam(self.std_model.parameters(), lr=0.001)
        self.use_min = use_min

    def __call__(
        self,
        replay_buffer,
        online_actor,
        target_actor,
        actor_optimizer,
        online_critic,
        target_critic,
        critic_optimizer,
        quantile_models,
        quantile_optimizers,
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

        match self.next_state_sample:
            case "unif":
                quantile_preds = quantile_models(states, actions)
                for i, tau in enumerate(quantile_models.quantiles):
                    loss_i = quantile_loss(quantile_preds[i], next_states, tau=tau)
                    quantile_optimizers[i].zero_grad()
                    loss_i.backward()
                    quantile_optimizers[i].step()
                candidate_next_states = sample_candidates_unif(
                    quantile_preds[0], quantile_preds[1], self.num_next_state_sample
                )
            case "triangular":
                quantile_preds = quantile_models(states, actions)
                for i, tau in enumerate(quantile_models.quantiles):
                    loss_i = quantile_loss(quantile_preds[i], next_states, tau=tau)
                    quantile_optimizers[i].zero_grad()
                    loss_i.backward()
                    quantile_optimizers[i].step()
                candidate_next_states = sample_candidates_triangular(
                    quantile_preds[0], quantile_preds[1], self.num_next_state_sample
                )
            case "null":
                candidate_next_states = next_states.unsqueeze(1)
            case "gaussian":
                predicted_log_std = self.std_model(states, actions)
                # Here we assume the "target" for the mean is the observed next state.
                # If the mean is provided by another network m, use: error = next_states - m.
                error = (
                    next_states - next_states
                )  # are not modeling a residual, you may need an alternative target.
                # In many cases, you’d have a learned mean and then train the std on the residual.
                gaussian_loss = gaussian_std_nll_loss(predicted_log_std, error)
                self.std_optimizer.zero_grad()
                gaussian_loss.backward()
                self.std_optimizer.step()

                candidate_next_states = sample_candidates_gaussian(
                    self.std_model,
                    states,
                    actions,
                    next_states,
                    self.num_next_state_sample,
                )

        bs, num_samples, _ = candidate_next_states.shape
        candidate_next_states_flat = candidate_next_states.reshape(
            bs * num_samples, self.state_dim
        )
        candidate_actions = target_actor(candidate_next_states_flat)
        candidate_qs = target_critic(candidate_next_states_flat, candidate_actions)
        candidate_qs = candidate_qs.reshape(bs, num_samples, 1)
        # Compute the target Q-value using the target networks.

        # true_next_state = next_states.unsqueeze(1)
        true_action = target_actor(next_states)
        true_q = target_critic(next_states, true_action).unsqueeze(1)

        if self.next_state_sample == "null":
            max_q = true_q.squeeze(1)
        else:
            all_candidate_qs = torch.cat([candidate_qs, true_q], dim=1)
            match self.use_min:
                case True:
                    max_q, _ = torch.min(all_candidate_qs, dim=1)
                case False:
                    max_q, _ = torch.min(all_candidate_qs, dim=1)

        target_q = rewards + gamma * (1 - dones) * max_q.detach()

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


class modelbasedrollouts:
    def __init__(
        self,
        state_dim,
        action_dim,
        batch_size,
        gamma=0.9,
        num_next_state_sample=10,
        next_state_sample: str = "unif",  # Options: "gaussian", "triangular", "unif", "null"
        use_min: bool = False,
        rollout_horizon: int = 5,
    ):
        self.batch_size = batch_size
        self.gamma = gamma
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_next_state_sample = num_next_state_sample
        self.next_state_sample = next_state_sample
        self.use_min = use_min
        self.rollout_horizon = rollout_horizon

        # Existing standard deviation model for the "gaussian" option.
        self.std_model = StdPredictor(state_dim=state_dim, action_dim=action_dim).to(
            DEVICE
        )
        self.std_optimizer = optim.Adam(self.std_model.parameters(), lr=0.001)

        # New reward predictor and its optimizer.
        self.reward_model = RewardPredictor(state_dim, action_dim).to(DEVICE)
        self.reward_optimizer = optim.Adam(self.reward_model.parameters(), lr=0.001)

    def multi_step_rollout_quantile(
        self, initial_states, target_actor, target_critic, gamma, quantile_models
    ):
        horizon = self.rollout_horizon
        batch_size = initial_states.shape[0]
        device = initial_states.device
        cumulative_reward = torch.zeros(batch_size, 1).to(device)
        discount = 1.0
        current_states = initial_states

        for step in range(horizon):
            actions = target_actor(current_states)
            # Use the appropriate branch for next state prediction.
            if self.next_state_sample == "unif":
                quantile_preds = quantile_models(current_states, actions)
                candidate_next_states = sample_candidates_unif(
                    quantile_preds[0], quantile_preds[1], num_samples=1
                )
            elif self.next_state_sample == "triangular":
                quantile_preds = quantile_models(current_states, actions)
                candidate_next_states = sample_candidates_triangular(
                    quantile_preds[0], quantile_preds[1], num_samples=1
                )
            elif self.next_state_sample == "null":
                candidate_next_states = current_states.unsqueeze(1)
            elif self.next_state_sample == "gaussian":
                predicted_log_std = self.std_model(current_states, actions)
                candidate_next_states = sample_candidates_gaussian(
                    self.std_model,
                    current_states,
                    actions,
                    None,
                    num_samples=1,
                )
            else:
                # Fallback: simply use the current state.
                candidate_next_states = current_states.unsqueeze(1)

            # We now have candidate_next_states of shape (batch_size, 1, state_dim).
            next_states_pred = candidate_next_states.squeeze(
                1
            )  # Shape: (batch_size, state_dim)
            # Predict the immediate reward for this simulated transition.
            predicted_rewards = self.reward_model(
                current_states, actions, next_states_pred
            )
            cumulative_reward += discount * predicted_rewards
            discount *= gamma
            current_states = next_states_pred

        # Bootstrapping: use the critic to estimate the value of the final state.
        final_actions = target_actor(current_states)
        final_q = target_critic(current_states, final_actions).detach()
        cumulative_reward += discount * final_q
        return cumulative_reward

    def __call__(
        self,
        replay_buffer,
        online_actor,
        target_actor,
        actor_optimizer,
        online_critic,
        target_critic,
        critic_optimizer,
        quantile_models,
        quantile_optimizers,
        **kwargs,
    ):
        gamma = self.gamma
        batch_size = self.batch_size

        if len(replay_buffer) < batch_size:
            return -1

        states_np, actions_np, rewards_np, next_states_np, dones_np = (
            replay_buffer.sample(batch_size)
        )
        states = torch.FloatTensor(states_np)
        actions = torch.FloatTensor(actions_np)
        rewards = torch.FloatTensor(rewards_np).unsqueeze(1)
        next_states = torch.FloatTensor(next_states_np)
        dones = torch.FloatTensor(dones_np).unsqueeze(1)

        # Update quantile models using one-step transitions.
        if self.next_state_sample in ["unif", "triangular"]:
            quantile_preds = quantile_models(states, actions)
            for i, tau in enumerate(quantile_models.quantiles):
                loss_i = quantile_loss(quantile_preds[i], next_states, tau=tau)
                quantile_optimizers[i].zero_grad()
                loss_i.backward()
                quantile_optimizers[i].step()
        elif self.next_state_sample == "gaussian":
            predicted_log_std = self.std_model(states, actions)
            error = next_states - next_states  # Zero error target
            gaussian_loss = gaussian_std_nll_loss(predicted_log_std, error)
            self.std_optimizer.zero_grad()
            gaussian_loss.backward()
            self.std_optimizer.step()

        # Update the reward model on actual transitions.
        predicted_rewards = self.reward_model(states, actions, next_states)
        reward_loss = F.mse_loss(predicted_rewards, rewards)
        self.reward_optimizer.zero_grad()
        reward_loss.backward()
        self.reward_optimizer.step()

        # Use a multi-step rollout with the quantile models to compute a synthetic target.
        synthetic_target = self.multi_step_rollout_quantile(
            states, target_actor, target_critic, gamma, quantile_models
        )

        # You might blend this synthetic multi-step return with a one-step target.
        # For simplicity, here we use the synthetic_target as the target Q value.
        target_q = synthetic_target * (1 - dones)

        # Critic update.
        current_q = online_critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q.detach())
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        # Actor update.
        actor_loss = -online_critic(states, online_actor(states)).mean()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        return actor_loss


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

        # Compute the target Q-value using the target networks.
        next_actions = target_actor(next_states)
        next_q = target_critic(next_states, next_actions)
        target_q = rewards + gamma * (1 - dones) * next_q.detach()

        # Critic update: minimize the MSE between current Q and target Q.
        current_q = online_critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        # Actor update: maximize the Q-value by minimizing the negative Q.
        actor_loss = -online_critic(states, online_actor(states)).mean()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        return actor_loss
