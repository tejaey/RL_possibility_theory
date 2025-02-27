from typing import Literal
import torch, random, math
from torch import FloatType, optim
import numpy as np
import torch.nn as nn
from abc import ABC, abstractmethod
import logging


def unpack_batch(batch):
    states, actions, rewards, next_states, dones = batch

    states_t = torch.FloatTensor(states)
    actions_t = torch.LongTensor(actions).unsqueeze(-1)
    rewards_t = torch.FloatTensor(rewards).unsqueeze(-1)
    next_states_t = torch.FloatTensor(next_states)
    dones_t = torch.FloatTensor(dones).unsqueeze(-1)

    return states_t, actions_t, rewards_t, next_states_t, dones_t


class td_loss_meta(ABC):
    def __init__(self):
        self.lossfunc = nn.MSELoss()

    @abstractmethod
    def __call__(self, batch, online_qnet, target_qnet, optimizers) -> float:
        pass


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
