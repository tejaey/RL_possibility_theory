from typing import Literal
import gymnasium
import torch.nn.functional as F
import torch, random, math
from torch import FloatType, optim
import numpy as np
import torch.nn as nn
from abc import ABC, abstractmethod
import logging

from qnets import (
    StdPredictor,
    QuantileModel,
    RewardModel,
    EnsembleQuantileModels,
    MeanStdPredictor,
)
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


class td_loss_ensemble_grad_updated2(td_loss_meta):
    """
    Attributes:
        ALPHA (float): smoothing factor for 'mle_max_update'
        GAMMA (float): reward discount
        BETA  (float): weight on normalized grad‑norm in L'
        use_ensemble_min (bool): whether to use the min over target ensemble for next Q
    """

    def __init__(
        self,
        GAMMA,
        ALPHA,
        BETA,
        possibility_update: Literal["mle", "mle_max_update"],
        use_ensemble_min: bool = False,
    ):
        super().__init__()
        self.GAMMA = GAMMA
        self.ALPHA = ALPHA
        self.BETA = BETA
        self.possibility_update = possibility_update
        self.use_ensemble_min = use_ensemble_min

    def __call__(self, batch, online_qnet, target_qnet, optimizers) -> float:
        # unpack (on DEVICE already)
        states_t, actions_t, rewards_t, next_states_t, dones_t = unpack_batch(batch)
        N = online_qnet.num_ensemble

        raw_losses = [0.0] * N
        grad_norms = [0.0] * N

        # 1) Compute raw TD-loss Li and gradient norms Gi, step once
        for i, qnet in enumerate(online_qnet.qnets):
            opt = optimizers[i]
            opt.zero_grad()

            # current Q
            cur_q = qnet(states_t).gather(1, actions_t)

            # next Q: either min over ensemble or per-net
            if self.use_ensemble_min:
                with torch.no_grad():
                    stacked = torch.stack(
                        [
                            tn(states_t).max(1, keepdim=True)[0]
                            for tn in target_qnet.qnets
                        ],
                        dim=0,
                    )
                    nxt_q = torch.amin(stacked, dim=0)
            else:
                with torch.no_grad():
                    nxt_q = target_qnet.qnets[i](next_states_t).max(1, keepdim=True)[0]

            # TD target and loss
            tgt_q = rewards_t + self.GAMMA * nxt_q * (1 - dones_t)
            Li = self.lossfunc(cur_q, tgt_q)
            Li.backward()

            # grad norm
            g2 = 0.0
            for p in qnet.parameters():
                if p.grad is not None:
                    g2 += p.grad.data.norm(2).item() ** 2
            Gi = math.sqrt(g2)

            opt.step()

            raw_losses[i] = Li.item()
            grad_norms[i] = Gi

        # 2) Normalize gradient norms
        maxG = max(grad_norms) + 1e-8

        # 3) Compute likelihoods via corrected losses L'
        likelihoods = []
        for Li, Gi in zip(raw_losses, grad_norms):
            Lp = Li + self.BETA * (Gi / maxG)
            likelihoods.append(math.exp(-Lp))

        # 4) Update possibilities
        poss = online_qnet.possibility
        if self.possibility_update == "mle":
            max_l = max(l * p for l, p in zip(likelihoods, poss)) + 1e-8
            for i in range(N):
                poss[i] = poss[i] * likelihoods[i] / max_l

        elif self.possibility_update == "mle_max_update":
            max_l = max(l * p for l, p in zip(likelihoods, poss)) + 1e-8
            for i in range(N):
                poss[i] = min(
                    max(self.ALPHA * poss[i], likelihoods[i] / max_l),
                    1.0,
                )
        else:
            raise ValueError(f"Unknown possibility_update {self.possibility_update!r}")

        # 5) Floor & normalize
        if max(poss) > 0:
            poss = [p / max(poss) for p in poss]
        online_qnet.possibility = [max(p, 0.001) for p in poss]

        return float(np.mean(raw_losses))


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


class td_loss_ensemble_grad_updated(td_loss_meta):
    """
    Attributes:
        ALPHA (float): smoothing factor for 'mle_max_update'
        GAMMA (float): reward discount
        BETA  (float): weight on normalized grad‐norm in L'
    """

    def __init__(
        self,
        GAMMA,
        ALPHA,
        BETA,
        possibility_update: Literal["mle", "mle_max_update"],
    ):
        super().__init__()
        self.GAMMA = GAMMA
        self.ALPHA = ALPHA
        self.BETA = BETA
        self.possibility_update = possibility_update

    def __call__(self, batch, online_qnet, target_qnet, optimizers) -> float:
        # 1) unpack and prepare
        states_t, actions_t, rewards_t, next_states_t, dones_t = unpack_batch(batch)
        N = online_qnet.num_ensemble

        raw_losses = [0.0] * N
        grad_norms = [0.0] * N

        # 2) single backward & step for each member, record Li and Gi
        for i, qnet in enumerate(online_qnet.qnets):
            opt = optimizers[i]
            opt.zero_grad()

            # forward pass
            current_q = qnet(states_t).gather(1, actions_t)
            with torch.no_grad():
                next_q = target_qnet.qnets[i](next_states_t).max(1, keepdim=True)[0]
            target_q = rewards_t + self.GAMMA * next_q * (1 - dones_t)

            # compute raw TD‐loss and backprop
            Li = self.lossfunc(current_q, target_q)
            Li.backward()

            # measure gradient norm Gi = ||∇θ Li||
            g2 = 0.0
            for p in qnet.parameters():
                if p.grad is not None:
                    g2 += p.grad.data.norm(2).item() ** 2
            Gi = math.sqrt(g2)

            # update parameters
            opt.step()

            raw_losses[i] = Li.item()
            grad_norms[i] = Gi

        # 3) normalize grad norms
        maxG = max(grad_norms) + 1e-8

        # 4) build likelihoods using corrected losses L'i
        likelihoods = []
        for Li, Gi in zip(raw_losses, grad_norms):
            Lp = Li + self.BETA * (Gi / maxG)
            likelihoods.append(math.exp(-Lp))

        # 5) update possibilities
        poss = online_qnet.possibility
        match self.possibility_update:
            case "mle":
                max_l = max(l * p for l, p in zip(likelihoods, poss)) + 1e-8
                for i in range(N):
                    poss[i] = poss[i] * likelihoods[i] / max_l

            case "mle_max_update":
                max_l = max(l * p for l, p in zip(likelihoods, poss)) + 1e-8
                for i in range(N):
                    # use ALPHA as a damping factor on old possibility
                    poss[i] = min(
                        max(self.ALPHA * poss[i], likelihoods[i] / max_l),
                        1.0,
                    )

            case _:
                raise ValueError(
                    f"Unknown possibility_update {self.possibility_update!r}"
                )

        # 6) floor and normalize
        # (first rescale so the max is 1, then floor at 0.001)
        max_poss = max(poss)
        if max_poss > 0:
            poss = [p / max_poss for p in poss]
        poss = [max(p, 0.001) for p in poss]

        online_qnet.possibility = poss
        return float(np.mean(raw_losses))


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
        next_state_sample: Literal["gaussian", "unif", "null"] = "unif",
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
                gaussian_loss = gaussian_std_nll_loss_bak(predicted_log_std, error)
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
            gaussian_loss = gaussian_std_nll_loss_bak(predicted_log_std, error)
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
        rollout_prob: float = 0.1,
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
        self.rollout_prob = 1 - rollout_prob
        self.epsilon = epsilon
        self.use_min = use_min

        # Reward model (deterministic)
        self.reward_model = RewardModel(state_dim).to(DEVICE)
        self.reward_opt = optim.Adam(self.reward_model.parameters(), lr=1e-3)

        # Imagination model
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
        states: torch.Tensor,  # (bs, state_dim)
        actions: torch.Tensor,  # (bs, action_dim)
        next_states: torch.Tensor,  # (bs, state_dim)
        target_actor: nn.Module,
        target_critic: nn.Module,
        rewards: torch.Tensor,  # (bs,1)
        dones: torch.Tensor,  # (bs,1)
    ) -> torch.Tensor:
        bs = states.size(0)
        γ = self.gamma

        # Standard TD(0) target
        with torch.no_grad():
            a1 = target_actor(next_states)  # (bs, action_dim)
            q1 = target_critic(next_states, a1)  # (bs, 1)
        y0 = (rewards + γ * (1 - dones) * q1).unsqueeze(1)  # (bs,1,1)

        # Train imagination model on real transitions
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

        # With probability rollout_prob, skip planning
        if random.random() < self.rollout_prob:
            return y0  # (bs,1,1)

        # Possibilistic planning (zero‐step)
        y_plan = []
        for _ in range(self.num_neighbour_sample):
            # Sample neighbour within ε‐ball
            eps = (
                torch.rand(bs, self.state_dim, device=states.device) * 2 - 1
            ) * self.epsilon
            s_star = states + eps  # (bs, state_dim)
            a_star = target_actor(s_star)  # (bs, action_dim)

            # Sample next‐state candidates and compute one‐step returns
            y_k_vals = []
            for _ in range(self.num_next_state_sample):
                if self.next_state_sample == "quantile":
                    ql, qu = self.quantile_models(s_star, a_star)
                    u = torch.rand_like(ql)
                    s_prime = (1 - u) * ql + u * qu
                else:  # Gaussian
                    mu, std = self.next_state_model(s_star, a_star)
                    s_prime = mu + std * torch.randn_like(std)

                # Predict reward and next‐step Q
                r_pred = self.reward_model(s_star, s_prime)  # (bs,1)
                a2 = target_actor(s_prime)  # (bs, action_dim)
                q2 = target_critic(s_prime, a2)  # (bs,1)

                y_i = r_pred + γ * q2  # (bs,1)
                y_k_vals.append(y_i)

            # Max over imagined samples
            y_k = torch.stack(y_k_vals, dim=1).max(dim=1, keepdim=True)[0]  # (bs,1)
            y_plan.append(y_k.unsqueeze(2))  # (bs,1,1)

        Y_plan = torch.cat(y_plan, dim=1)  # (bs, K, 1)
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

        # Sample transition batch
        s, a, r, s_next, done = replay_buffer.sample(self.batch_size)
        states = torch.FloatTensor(s).to(DEVICE)
        actions = torch.FloatTensor(a).to(DEVICE)
        rewards = torch.FloatTensor(r).unsqueeze(-1).to(DEVICE)
        next_states = torch.FloatTensor(s_next).to(DEVICE)
        dones = torch.FloatTensor(done).unsqueeze(-1).to(DEVICE)

        # Train reward model on real data
        r_pred = self.reward_model(states, next_states)
        loss_r = F.mse_loss(r_pred, rewards)
        self.reward_opt.zero_grad()
        loss_r.backward()
        self.reward_opt.step()

        # Compute targets (real or imagined)
        Y = self.compute_yj(
            states, actions, next_states, target_actor, target_critic, rewards, dones
        )  # shape = (bs, N, 1) or (bs,1,1)
        if self.use_min:
            Y_agg, _ = torch.min(Y, dim=1)
        else:
            Y_agg, _ = torch.max(Y, dim=1)
        target_q = Y_agg.detach()  # (bs,1)

        # Critic update
        current_q = online_critic(states, actions)
        loss_c = F.mse_loss(current_q, target_q)
        critic_optimizer.zero_grad()
        loss_c.backward()
        critic_optimizer.step()

        # Actor update
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
        rollout_prob: float = 0.1,
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
        self.rollout_prob = 1 - rollout_prob
        # 1) Learned reward model R(s, s')
        self.reward_model = RewardModel(state_dim).to(DEVICE)
        self.reward_opt = optim.Adam(self.reward_model.parameters(), lr=1e-3)

        # 2) Imagination model (only if rollout_depth=1 and not "null")
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
        # else: no predictor needed for depth=0 or "null"

    def compute_yj(
        self,
        states: torch.Tensor,  # (bs, state_dim)
        actions: torch.Tensor,  # (bs, action_dim)
        next_states: torch.Tensor,  # (bs, state_dim)
        target_actor: nn.Module,
        target_critic: nn.Module,
        rewards: torch.Tensor,  # (bs,1)
        dones: torch.Tensor,  # (bs,1)
    ) -> torch.Tensor:
        bs = states.size(0)
        γ = self.gamma

        # --- Step 0: Standard TD(0) bootstrap on recorded transition ---
        with torch.no_grad():
            a1 = target_actor(next_states)  # (bs, A)
            q1 = target_critic(next_states, a1)  # (bs, 1)
        # shape (bs, 1, 1)
        y0 = (rewards + γ * (1 - dones) * q1).unsqueeze(1)

        if self.rollout_depth == 0 or self.next_state_sample == "null":
            return y0  # (bs,1,1)

        # --- Now rollout_depth == 1 and next_state_sample != "null" ---
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

        if random.random() < self.rollout_prob:
            return y0
        # --- One imagined step (all in no_grad) ---
        with torch.no_grad():
            # expand recorded next_state s1 to shape (bs, N, D)
            N = self.num_next_state_sample
            s1 = next_states.unsqueeze(1).expand(bs, N, self.state_dim)  # (bs,N,D)
            a1_rep = target_actor(s1.reshape(-1, self.state_dim))  # (bs*N, A)

            # generate s2 candidates
            if self.next_state_sample == "quantile":
                ql, qu = self.quantile_models(s1.reshape(-1, self.state_dim), a1_rep)
                eps = torch.rand_like(ql)
                s2_flat = (1 - eps) * ql + eps * qu
            else:  # gaussian
                mu, std = self.next_state_model(s1.reshape(-1, self.state_dim), a1_rep)
                s2_flat = mu + std * torch.randn_like(std)

            # reshape s2_flat → (bs, N, D)
            s2 = s2_flat.view(bs, N, self.state_dim)

            # learned reward R(s1, s2)
            r1 = self.reward_model(s1.reshape(-1, self.state_dim), s2_flat).view(
                bs, N, 1
            )

            # Q(s2, μ(s2))
            a2 = target_actor(s2_flat)  # (bs*N, A)
            q2 = target_critic(s2_flat, a2).view(bs, N, 1)  # (bs,N,1)

            # one imagined Bellman step: R + γ Q
            imag_term = r1 + γ * q2  # (bs,N,1)

        # final y1: r0 + γ * imag_term  => r0 + γ*R + γ^2 Q
        y1 = rewards.unsqueeze(1) + γ * imag_term  # (bs,N,1)

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

        # 1) sample batch
        s, a, r, s_next, done = replay_buffer.sample(self.batch_size)
        states = torch.FloatTensor(s).to(DEVICE)
        actions = torch.FloatTensor(a).to(DEVICE)
        rewards = torch.FloatTensor(r).unsqueeze(-1).to(DEVICE)
        next_states = torch.FloatTensor(s_next).to(DEVICE)
        dones = torch.FloatTensor(done).unsqueeze(-1).to(DEVICE)

        # 2) train reward model on real (s,s')
        r_pred = self.reward_model(states, next_states)
        loss_r = F.mse_loss(r_pred, rewards)
        self.reward_opt.zero_grad()
        loss_r.backward()
        self.reward_opt.step()

        # 3) compute y_j (0 or 1 step)
        Y = self.compute_yj(
            states, actions, next_states, target_actor, target_critic, rewards, dones
        )
        # Y.shape = (bs, N, 1)
        if self.use_min:
            Y_agg, _ = torch.min(Y, dim=1)  # (bs,1)
        else:
            Y_agg, _ = torch.max(Y, dim=1)  # (bs,1)

        # 4) critic update
        target_q = Y_agg.detach()
        current_q = online_critic(states, actions)
        loss_c = F.mse_loss(current_q, target_q)
        critic_optimizer.zero_grad()
        loss_c.backward()
        critic_optimizer.step()

        # 5) actor update
        loss_a = -online_critic(states, online_actor(states)).mean()
        actor_optimizer.zero_grad()
        loss_a.backward()
        actor_optimizer.step()

        return loss_a
