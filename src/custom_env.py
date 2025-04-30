import gymnasium as gym
from gymnasium.envs.mujoco.walker2d_v5 import Walker2dEnv
from gymnasium.envs.mujoco.half_cheetah_v5 import HalfCheetahEnv
from gymnasium.envs.mujoco.hopper_v5 import HopperEnv
from gymnasium.envs.box2d.lunar_lander import LunarLander
from gymnasium.envs.classic_control.cartpole import CartPoleEnv
from gymnasium import spaces
from gymnasium.utils import seeding
import random
import numpy as np


class SparseWalker2DEnv(Walker2dEnv):
    def step(self, action):
        obs, reward, terminated, truncated, _ = super().step(action)
        # removes the alive bonus for walker2d_v5
        reward -= 1.0
        return obs, reward, terminated, truncated, _


class SparseHopper(HopperEnv):
    def _get_rew(self, x_velocity: float, action):
        forward_reward = self._forward_reward_weight * x_velocity
        healthy_reward = self.healthy_reward
        rewards = forward_reward + healthy_reward

        ctrl_cost = self.control_cost(action)
        costs = ctrl_cost

        reward = rewards - costs

        reward_info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_survive": healthy_reward,
        }

        # only gets forward rewards
        return forward_reward, reward_info


class SparseHalfCheetah(HalfCheetahEnv):
    def _get_rew(self, x_velocity: float, action):
        forward_reward = self._forward_reward_weight * x_velocity

        reward = forward_reward

        reward_info = {"reward_forward": forward_reward, "reward_ctrl": 0}
        ## only gets forward_reward, no control cost loss

        return reward, reward_info


class SparseCartPole(CartPoleEnv):
    """
    CartPole environment where reward is only given at the end of the episode.
    +1 if the pole survives for the maximum duration, 0 otherwise.
    """

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        # Standard reward is +1 for every step
        # Make it sparse: only give reward if the episode *ends* successfully (not terminated)
        if truncated:  # Episode reached max steps without failure
            sparse_reward = 1.0
        elif terminated:  # Episode failed (pole fell)
            sparse_reward = 0.0
        else:  # Episode continues
            sparse_reward = 0.0

        return obs, sparse_reward, terminated, truncated, info


# class SparseLunarLander(LunarLander):
#     """
#     LunarLander environment where reward is sparse.
#     +1 for successful landing, -1 for crash, 0 otherwise.
#     Removes shaping rewards for legs, engine firing etc.
#     """
#
#     def step(self, action):
#         obs, reward, terminated, truncated, info = super().step(action)
#
#         # Original reward includes shaping terms.
#         # Let's make it sparse based only on the final outcome.
#         sparse_reward = 0.0
#         if terminated:
#             if self.game_over:  # Crashed
#                 sparse_reward = -1.0
#             elif (
#                 self.lander.awake
#             ):  # Landed successfully (check if awake is the right flag)
#                 # Need to be careful: 'terminated' might be true just from landing.
#                 # Check velocity conditions if needed for "successful" landing
#                 # For simplicity, let's assume terminated + not crashed = landed
#                 sparse_reward = 1.0
#         # No reward if truncated or episode continues without termination
#
#         return obs, sparse_reward, terminated, truncated, info
#


class SparseLunarLander(LunarLander):
    """
    LunarLander environment with minimal shaping:
      • Sparse terminal reward: +1 for landing, -1 for crash
      • Shaping during episode:
        – small penalty proportional to distance from pad
        – small penalty for horizontal/vertical speed
    """

    def step(self, action):
        obs, _, terminated, truncated, info = super().step(action)

        # Unpack state
        x, y, vx, vy, *_ = obs

        # Sparse shaping
        reward = 0.0
        reward -= 0.04 * np.sqrt(x**2 + y**2)  # distance penalty
        reward -= 0.02 * (abs(vx) + abs(vy))  # velocity penalty

        # Terminal reward
        if terminated:
            reward = 1.0 if self.lander.awake else -1.0

        return obs, reward, terminated, truncated, info


class StochasticCartPole(CartPoleEnv):
    def __init__(self, noise_scale: float = 0.3, **kwargs):
        super().__init__(**kwargs)
        self.noise_scale = noise_scale

    def step(self, action):
        # perform the normal transition
        obs, reward, terminated, truncated, info = super().step(action)
        # add small Gaussian noise to each state variable
        noise = np.random.normal(loc=0.0, scale=self.noise_scale, size=obs.shape)
        obs_noisy = obs + noise
        return obs_noisy, reward, terminated, truncated, info


def make_env(id: str, **kwargs) -> gym.Env:
    match id:
        case "SparseHalfCheetah":
            return SparseHalfCheetah()
        case "SparseWalker2DEnv":
            return SparseWalker2DEnv()
        case "SparseHopper":
            return SparseHopper()
        case "SparseCartPole":
            return SparseCartPole(**kwargs)
        case "SparseLunarLander":
            return SparseLunarLander(**kwargs)
        case "StochasticCartPole":
            return StochasticCartPole(**kwargs)
        case _:
            return gym.make(id=id, **kwargs)
