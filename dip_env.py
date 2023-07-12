import numpy as np
import time
import sys

import gymnasium as gym

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines

from math import cos, sin
from scipy import signal
from PIL import Image
from io import BytesIO
from gym import spaces, logger
from gym.utils import seeding
import math
from scipy.integrate import ode


# Enable interactive mode
plt.ion()


class DoubleInvertedPendulumCartEnv(gym.Env):
    def __init__(self, episode_len, action_scaler=1, render_mode=None):
        self.cart_mass = 0.5
        self.pendulum_mass_1 = 0.162
        self.pendulum_mass_2 = 0.203
        self.total_mass = self.cart_mass + self.pendulum_mass_1 + self.pendulum_mass_2
        self.pendulum_length_1 = 0.170
        self.pendulum_cg_1 = 0.085
        self.pendulum_length_2 = 0.314
        self.pendulum_cg_2 = 0.157
        self.gravity = 9.81
        self.inertia_1 = (self.pendulum_mass_1 * (self.pendulum_length_1**2))/12
        self.inertia_2 = (self.pendulum_mass_2 * (self.pendulum_length_2**2))/12
        self.tau = 0.0395

        self.episode_len = episode_len
        self.action_scaler = action_scaler
        self.render_mode = render_mode

        self.x_goal_position = 0

        self.viewer = None
        self.scale = 100  # Scale factor for converting coordinates to pixels
        # self.seed()
        # self.state = None #arr2mat
        self.steps_beyond_done = None
        self.x_threshold = 1.5 #1.0

        # Observation space and limits
        self.x_min = -(self.x_threshold * 2)
        self.x_max = (self.x_threshold * 2)
        self.theta_min = -np.finfo(np.float32).max
        self.theta_max = np.finfo(np.float32).max
        self.velocity_min = -np.finfo(np.float32).max
        self.velocity_max = np.finfo(np.float32).max

        self.min_observation = np.array([
            self.velocity_min, self.velocity_min, self.theta_min, self.theta_min,
            self.x_min, self.velocity_min], dtype=np.float32)

        self.max_observation = np.array([
            self.velocity_max, self.velocity_max, self.theta_max, self.theta_max,
            self.x_max, self.velocity_max], dtype=np.float32)

        self.observation_space = spaces.Box(low=self.min_observation,
                                            high=self.max_observation,
                                            dtype=np.float32)
        self.min_action = -1  # min cart force, min rope force
        self.max_action = 1  # max cart force, max rope force

        self.action_space = spaces.Box(low=self.min_action,
                                       high=self.max_action,
                                       shape=(1,),
                                       dtype=np.float32)

        # Define other necessary attributes
        self.x_goal_position = 0
        self.tau1 = 0.02  # Time step for the simulation
        self.counter = 0  # Taken from fregu856

        # Constants for the dynamic model
        self.h1 = self.cart_mass + self.pendulum_mass_1 + self.pendulum_mass_2
        self.h2 = (self.pendulum_mass_1 * self.pendulum_cg_1 +
                   self.pendulum_mass_2 * self.pendulum_length_1)
        self.h3 = self.pendulum_mass_2 * self.pendulum_cg_2
        self.h4 = (self.pendulum_mass_1 * (self.pendulum_cg_1**2) +
                   self.pendulum_mass_2 * (self.pendulum_length_1**2) + self.inertia_1)
        self.h5 = self.pendulum_mass_2 * self.pendulum_cg_2 * self.pendulum_length_1
        self.h6 = self.pendulum_mass_2 * (self.pendulum_cg_2**2) + self.inertia_2
        self.h7 = (self.pendulum_mass_1 * self.pendulum_cg_1 * self.gravity +
                   self.pendulum_mass_2 * self.pendulum_length_1 * self.gravity)
        self.h8 = self.pendulum_mass_2 * self.pendulum_cg_2 * self.gravity

        self.max_x_vel = -np.inf
        self.max_joint_vel = -np.inf
        self.vel_threshold = 300#200#40.0#35.0
        self.normalizer = np.array([self.vel_threshold, self.vel_threshold,
                                    np.pi, np.pi,
                                    self.x_threshold, 1])

    def step(self, action):
        state = self.state
        action = np.clip(action, -1.0, 1.0) * self.action_scaler

        u = action
        self.counter += 1

        def func(state, u):
            theta_dot = state.item(0)
            phi_dot = state.item(1)
            theta = state.item(2)
            phi = state.item(3)
            x = state.item(4)
            x_dot = state.item(5)
            state = np.array([[theta_dot], [phi_dot], [theta], [phi], [x], [x_dot]])

        # Constants for calculating a(Q) and B(Q)
            M11 = np.matrix([[self.h4, self.h5 * np.cos(theta - phi)],
                             [self.h5 * np.cos(theta - phi), self.h6]])
            M12 = np.matrix([[self.h2 * np.cos(theta)], [self.h3 * np.cos(phi)]])
            C1 = np.matrix([[self.h5 * np.sin(theta - phi) * ((phi_dot)**2)],
                            [-self.h5 * np.sin(theta - phi) * ((theta_dot) ** 2)]])
            G1 = np.matrix([[-self.h7 * np.sin(theta)], [-self.h8 * np.sin(phi)]])

            D = np.matrix(
                [[self.h1, self.h2 * np. cos(theta), self.h3 * np.cos(phi)],
                 [self.h2 * np.cos(theta), self.h4, self.h5 * np.cos(theta-phi)],
                 [self.h3 * np.cos(phi), self.h5 * np.cos(theta-phi), self.h6]])

            G = np.matrix([[0], [-self.h7 * np.sin(theta)], [-self.h8 * np.sin(phi)]])

            H = np.matrix([[1], [0], [0]])

            I = np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

            O_3_3 = np.matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

            O_3_1 = np.matrix([[0], [0], [0]])

            M11_inv = np.linalg.inv(M11)
            M12_mul = np.dot(M12, (-x_dot / self.tau))
            C1_G1_sum = C1 + G1
            MCG = (M12_mul + C1_G1_sum)
            result1 = np.dot(-M11_inv, MCG)  # matrix calculation in A matrix
            M12_div = (M12 / self.tau)
            result2 = np.dot(-M11_inv, M12_div)  # matrix calculation in B matrix

            # W = np.bmat([[O_3_1],[np.linalg.inv(D)*G]])

            a_Q = np.matrix(
                [[result1[0, 0]], [result1[1, 0]], [theta_dot], [phi_dot], [x_dot],
                 [-x_dot / self.tau]])
            B_Q = np.matrix(
                [[result2[0, 0]], [result2[1, 0]], [0], [0], [0], [1/self.tau]])
            Q_dot = a_Q + B_Q * u
            return Q_dot

        state_dot = func(state, u)
        state_dot_new = np.matrix(state_dot)

        self.state = (np.reshape(self.state, (6, 1)) + self.tau1 * state_dot_new)
        self.state = np.squeeze(self.state)

        # flag = self.x_goal_position

        theta_dot = self.state.item(0)
        phi_dot = self.state.item(1)
        theta = normalize_angle(self.state.item(2))
        phi = normalize_angle(self.state.item(3))
        x = state.item(4)
        x_dot = state.item(5)

        # angle_reward = (np.exp(-(theta**2)) + np.exp(-(phi**2))) / 2
        # angle_reward = np.exp(-max(theta**2, phi**2))
        angle_reward = np.exp(-(theta+phi))

        reward = angle_reward

        # y_tip = (self.pendulum_length_1 * np.cos(theta) +
        #          self.pendulum_length_2 * np.cos(phi))

        # y_tip_reward = np.exp(-5*np.abs(y_tip - 0.484))
        # reward = y_tip_reward

        terminated = bool(x < -self.x_threshold or x > self.x_threshold)
        terminated = terminated or bool(np.abs(theta_dot) > self.vel_threshold or
                                        np.abs(phi_dot) > self.vel_threshold)

        truncated = bool(self.counter >= self.episode_len)

        if self.render_mode == 'human':
            img = self.render()
            info = {'img': img}
        else:
            info = {}

        # x_tip = (x + self.pendulum_length_1 * np.sin(theta) +
        #          self.pendulum_length_2 * np.sin(phi))
        # y_tip = (self.pendulum_length_1 * np.cos(theta) +
        #          self.pendulum_length_2 * np.cos(phi))

        # print(f"action: {np.round(action[0], 4)}  reward: {np.round(reward, 4)}  " +
        #       f"theta_dot: {np.round(theta_dot, 4)}  phi_dot: {np.round(phi_dot, 4)}  " +
        #       f"theta: {np.round(self.state.item(2), 4)}  phi: {np.round(self.state.item(3), 4)}")
        # input()

        # self.max_x_vel = max(self.max_x_vel, self.state[0, 5])
        # self.max_joint_vel = max(self.max_joint_vel, max(theta_dot, phi_dot))
        # if terminated or truncated:
        #     print(f"max_x_vel: {self.max_x_vel} -- max_joint_vel: {self.max_joint_vel}")

        # Normalize state
        # norm_state = \
        #     np.array([self.state.item(0), self.state.item(1), theta, phi,
        #               self.state.item(4), self.state.item(5)]) / self.normalizer
        norm_state = \
            np.array([theta_dot, phi_dot, theta, phi, x, x_dot]) / self.normalizer

        return norm_state, reward, terminated, truncated, info

    def reset(self, seed=None, fixed_init=False):
        if seed is not None:
            np.random.seed(seed)

        # Reset the state
        self.state = np.array([
            np.random.normal(loc=0, scale=0.1),  # theta_dot
            np.random.normal(loc=0, scale=0.1),  # phi_dot
            # np.random.uniform(low=-np.pi, high=np.pi),  # theta
            # np.random.uniform(low=-np.pi, high=np.pi),  # phi
            np.random.uniform(low=-np.pi-0.314, high=-np.pi+0.314),  # theta
            np.random.uniform(low=-np.pi-0.314, high=-np.pi+0.314),  # phi
            # 0, 0,
            # -np.pi, -np.pi,
            np.random.uniform(low=-0.1, high=0.1),  # x
            np.random.normal(loc=0, scale=0.1),  # x_dot
        ])

        self.steps_beyond_done = None
        self.counter = 0

        return self.state, {}

    def render(self, return_image=True):
        plt.cla()
        plt.xlim(-2, 2)
        plt.ylim(-1, 1)

        cart_width = 0.02
        cart_height = 0.02
        cart_x = float(self.state[0, 4])
        cart_y = float(0)

        pendulum_1_x = float(cart_x + self.pendulum_length_1 * np.sin(self.state[0, 2]))
        pendulum_1_y = float(cart_y + self.pendulum_length_1 * np.cos(self.state[0, 2]))

        pendulum_2_x = float(
            pendulum_1_x + self.pendulum_length_2 * np.sin(self.state[0, 3]))
        pendulum_2_y = float(
            pendulum_1_y + self.pendulum_length_2 * np.cos(self.state[0, 3]))

        cart_x_left = cart_x - cart_width/2
        cart_x_right = cart_x + cart_width/2
        cart_y_top = cart_y + cart_height/2
        cart_y_bottom = cart_y - cart_height/2

        rect_vertices = np.array([[cart_x_left, cart_y_bottom],
                                 [cart_x_right, cart_y_bottom],
                                 [cart_x_right, cart_y_top],
                                 [cart_x_left, cart_y_top],
                                 [cart_x_left, cart_y_bottom]])

        # Plot the rectangular patch
        # Connect vertices to form rectangle
        plt.plot(rect_vertices[:, 0], rect_vertices[:, 1], 'k-')
        # Plot a marker at the center of the rectangle
        plt.plot(cart_x, cart_y, 'ko', markersize=10)

        plt.plot([cart_x, pendulum_1_x], [cart_y, pendulum_1_y], 'r-')
        plt.plot([pendulum_1_x, pendulum_2_x], [pendulum_1_y, pendulum_2_y], 'b-')
        plt.plot(pendulum_1_x, pendulum_1_y, 'ro', markersize=5)
        plt.plot(pendulum_2_x, pendulum_2_y, 'bo', markersize=5)

        plt.pause(0.01)
        plt.draw()

        if return_image:
            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img = Image.open(buf)
            return img

    def close(self):
        plt.close()

# frames = []
# env = DoubleInvertedPendulumCartEnv()
# frame = env.render(return_image=True)
#     #print(f"Step: {i}, New State: {obs}")  # Add this print statement
# if frame is not None:
#     frames.append(frame)
#     time.sleep(0.0001)


def normalize_angle(angle):
    """
    3*pi gives -pi, 4*pi gives 0 etc, etc. (returns the negative difference
    from the closest multiple of 2*pi)
    """
    normalized_angle = abs(angle)
    normalized_angle = normalized_angle % (2*np.pi)
    if normalized_angle > np.pi:
        normalized_angle = normalized_angle - 2*np.pi
    normalized_angle = abs(normalized_angle)
    return normalized_angle
