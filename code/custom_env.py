import gymnasium as gym
from gymnasium.envs.mujoco.walker2d_v5 import Walker2dEnv
from gymnasium.envs.mujoco.half_cheetah_v5 import HalfCheetahEnv
from gymnasium.envs.mujoco.hopper_v5 import HopperEnv
from gymnasium import spaces
from gymnasium.utils import seeding
import random
import numpy as np
# from gymnasium.envs.classic_control import rendering


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    def __init__(self):
        super().__init__()
        # Discrete actions: 0=north, 1=east, 2=south, 3=west
        self.action_space = spaces.Discrete(4)

        # Observation as a tuple (row, col)
        self.observation_space = spaces.Tuple((spaces.Discrete(3), spaces.Discrete(4)))
        self.rewards = [[0, 0, 0, 1], [0, 0, 0, -1], [0, 0, 0, 0]]
        self.max_r = 2
        self.max_c = 3
        self.rock = (1, 1)
        self.viewer = None
        self.state = None
        self.seed()

    def check_valid_step(self, action):
        r, c = self.state
        if action == 0 and r == 0:
            return -1
        if action == 1 and c == self.max_c:
            return -1
        if action == 2 and r == self.max_r:
            return -1
        if action == 3 and c == 0:
            return -1
        return action

    def step(self, action):
        # Gymnasium's step API: returns (observation, reward, terminated, truncated, info)
        terminated = False
        truncated = False  # For simplicity, we are not using truncation here.
        r, c = self.state
        coinflip = random.random()
        right_angle = 0
        if coinflip < 0.1:
            right_angle = 1
        elif coinflip > 0.9:
            right_angle = 2

        # Rotate action by 90° left or right based on coinflip.
        if right_angle == 1:
            action = (action - 1) % 4
        elif right_angle == 2:
            action = (action + 1) % 4

        # Check if the action is legal.
        valid_action = self.check_valid_step(action)
        if valid_action == -1:
            new_r, new_c = r, c
        else:
            if valid_action == 0:
                new_r, new_c = r - 1, c
            elif valid_action == 1:
                new_r, new_c = r, c + 1
            elif valid_action == 2:
                new_r, new_c = r + 1, c
            elif valid_action == 3:
                new_r, new_c = r, c - 1

        # If the new location is a rock, remain in the same position.
        if (new_r, new_c) == self.rock:
            new_r, new_c = r, c

        # Terminate if the agent reaches one of the goal states.
        if (new_r, new_c) == (0, 3) or (new_r, new_c) == (1, 3):
            terminated = True

        self.state = (new_r, new_c)
        reward = self.rewards[new_r][new_c]
        observation = np.array(self.state, dtype=np.int32)
        return observation, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        self.seed(seed)
        self.state = (2, 3)
        observation = np.array(self.state, dtype=np.int32)
        return observation, {}

    def render(self, mode="human"):
        screen_width = 400
        screen_height = 300
        robot_height = 20
        robot_width = 10
        offset = 50

        r, c = self.state
        # Coordinates for grid cell center (starts from bottom-left)
        grid_center_x = (c + 1) * 100 - offset
        grid_center_y = (self.max_r - r + 1) * 100 - offset

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)
            # Draw rock.
            rock = rendering.FilledPolygon(
                [
                    (1 * 100, 1 * 100),
                    (2 * 100, 1 * 100),
                    (2 * 100, 2 * 100),
                    (1 * 100, 2 * 100),
                ]
            )
            rock.set_color(0.8, 0.6, 0.4)
            # Draw green goal area.
            green = rendering.FilledPolygon(
                [
                    (3 * 100, 1 * 100),
                    (3 * 100, 2 * 100),
                    (4 * 100, 2 * 100),
                    (4 * 100, 1 * 100),
                ]
            )
            green.set_color(0, 1, 0)
            # Draw red goal area.
            red = rendering.FilledPolygon(
                [
                    (3 * 100, 2 * 100),
                    (3 * 100, 3 * 100),
                    (4 * 100, 3 * 100),
                    (4 * 100, 2 * 100),
                ]
            )
            red.set_color(1, 0, 0)
            self.viewer.add_geom(rock)
            self.viewer.add_geom(green)
            self.viewer.add_geom(red)

        # Draw the agent (robot) on the grid.
        l = grid_center_x - robot_width / 2
        r_val = grid_center_x + robot_width / 2
        t = grid_center_y + robot_height / 2
        b = grid_center_y - robot_height / 2
        robot = rendering.FilledPolygon([(l, b), (l, t), (r_val, t), (r_val, b)])
        self.viewer.add_onetime(robot)

        # Draw grid lines.
        self.viewer.draw_line((0, 100), (screen_width, 100))
        self.viewer.draw_line((0, 200), (screen_width, 200))
        self.viewer.draw_line((100, 0), (100, screen_height))
        self.viewer.draw_line((200, 0), (200, screen_height))
        self.viewer.draw_line((300, 0), (300, screen_height))

        return self.viewer.render(return_rgb_array=(mode == "rgb_array"))

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


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


def make_env(id: str, **kwargs) -> gym.Env:
    match id:
        case "SparseHalfCheetah":
            return SparseHalfCheetah()
        case "SparseWalker2DEnv":
            return SparseWalker2DEnv()
        case "SparseHopper":
            return SparseHopper()
        case _:
            return gym.make(id=id, **kwargs)
