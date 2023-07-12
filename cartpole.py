import logging
import math
import gym
from gym import spaces
import numpy as np
from scipy.integrate import ode

import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO


sin = np.sin
cos = np.cos


class DIPCEnv(gym.Env):
    def __init__(self, episode_len, action_scaler=1, render_mode=None):
        # # Original
        # self.g = -9.81  # gravity constant
        # self.m0 = 1.0  # mass of cart
        # self.m1 = 0.5  # mass of pole 1
        # self.m2 = 0.5  # mass of pole 2
        # self.L1 = 1  # length of pole 1
        # self.L2 = 1  # length of pole 2
        # self.tau = 0.02  # seconds between state updates
        # self.x_threshold = 2.4
        # self.plt_xlim = (-3, 3,)
        # self.plt_ylim = (-2.5, 2.5,)

        # George's
        self.g = -9.81 #???? # gravity constant
        self.m0 = 0.5  # mass of cart
        self.m1 = 0.162  # mass of pole 1
        self.m2 = 0.203  # mass of pole 2
        self.L1 = 0.170  # length of pole 1
        self.L2 = 0.314  # length of pole 2
        self.tau = 0.0395  # seconds between state updates
        self.x_threshold = 1.0
        self.plt_xlim = (-2, 2,)
        self.plt_ylim = (-1, 1,)

        self.l1 = self.L1/2  # distance from pivot point to center of mass
        self.l2 = self.L2/2  # distance from pivot point to center of mass
        self.I1 = self.m1 * (self.L1**2)/12  # moment of inertia of pole 1 w.r.t its COM
        self.I2 = self.m2 * (self.L2**2)/12  # moment of inertia of pole 2 w.r.t its COM

        self.d1 = self.m0 + self.m1 + self.m2
        self.d2 = (self.m1 * self.l1) + (self.m2 * self.L1)
        self.d3 = self.m2 * self.l2
        self.d4 = self.m1 * pow(self.l1, 2) + self.m2 * pow(self.L1, 2) + self.I1
        self.d5 = self.m2 * self.L1 * self.l2
        self.d6 = self.m2 * pow(self.l2, 2) + self.I2
        self.f1 = (self.m1 * self.l1 + self.m2 * self.L1)*self.g
        self.f2 = self.m2 * self.l2 * self.g

        self.counter = 0
        self.episode_len = episode_len
        self.action_scaler = action_scaler
        self.render_mode = render_mode

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

        self.vel_threshold = 1#300#200#40.0#35.0
        self.normalizer = np.array([self.x_threshold, np.pi, np.pi,
                                    1, self.vel_threshold, self.vel_threshold])

    def step(self, action):
        state = self.state
        action = np.clip(action, -1.0, 1.0) * self.action_scaler

        u = action
        self.counter += 1

        # (state_dot = func(state))
        def func(t, state, u):
            x = state.item(0)
            theta = state.item(1)
            phi = state.item(2)
            x_dot = state.item(3)
            theta_dot = state.item(4)
            phi_dot = state.item(5)

            # this is needed for some weird reason
            state = np.matrix([[x], [theta], [phi], [x_dot], [theta_dot], [phi_dot]])

            D = np.matrix(
                [[self.d1, self.d2 * cos(theta), self.d3 * cos(phi)],
                 [self.d2 * cos(theta), self.d4, self.d5 * cos(theta-phi)],
                 [self.d3 * cos(phi), self.d5 * cos(theta-phi), self.d6]])

            C = np.matrix(
                [[0, -self.d2 * sin(theta) * theta_dot, -self.d3 * sin(phi)*phi_dot],
                 [0, 0, self.d5 * sin(theta-phi) * phi_dot],
                 [0, -self.d5 * sin(theta-phi) * theta_dot, 0]])

            G = np.matrix([[0], [-self.f1 * sin(theta)], [-self.f2 * sin(phi)]])

            H = np.matrix([[1], [0], [0]])

            I_mat = np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            O_3_3 = np.matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
            O_3_1 = np.matrix([[0], [0], [0]])

            A_tilde = np.bmat([[O_3_3, I_mat], [O_3_3, -np.linalg.inv(D)*C]])
            B_tilde = np.bmat([[O_3_1], [np.linalg.inv(D)*H]])
            W = np.bmat([[O_3_1], [np.linalg.inv(D)*G]])
            state_dot = A_tilde*state + B_tilde*u + W
            return state_dot

        solver = ode(func)
        solver.set_integrator("dop853")  # (Runge-Kutta)
        solver.set_f_params(u)
        t0 = 0
        state0 = state
        solver.set_initial_value(state0, t0)
        solver.integrate(self.tau)
        state = solver.y
        self.state = state

        x = state.item(0)
        theta = normalize_angle(state.item(1))
        phi = normalize_angle(state.item(2))
        x_dot = state.item(3)
        theta_dot = state.item(4)
        phi_dot = state.item(5)

        angle_reward = np.exp(-(theta+phi))
        reward = angle_reward

        terminated = bool(x < -self.x_threshold or x > self.x_threshold)
        # terminated = terminated or bool(np.abs(theta_dot) > self.vel_threshold or
        #                                 np.abs(phi_dot) > self.vel_threshold)

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
        #       f"theta: {np.round(theta, 4)}  phi: {np.round(phi, 4)}")
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
            np.array([x, theta, phi, x_dot, theta_dot, phi_dot]) / self.normalizer

        return norm_state, reward, terminated, truncated, info

    def reset(self, seed=None, fixed_init=False):
        if seed is not None:
            np.random.seed(seed)

        # Reset the state
        self.state = np.array([
            np.random.uniform(low=-0.1, high=0.1),  # x
            # np.random.uniform(low=-np.pi, high=np.pi),  # theta1
            # np.random.uniform(low=-np.pi, high=np.pi),  # phi
            np.random.uniform(low=-np.pi-0.314, high=-np.pi+0.314),  # theta
            np.random.uniform(low=-np.pi-0.314, high=-np.pi+0.314),  # phi
            # 0, 0,
            # -np.pi, -np.pi,
            np.random.normal(loc=0, scale=0.1),  # x_dot
            np.random.normal(loc=0, scale=0.1),  # theta_dot
            np.random.normal(loc=0, scale=0.1),  # phi_dot
        ])

        self.steps_beyond_done = None
        self.counter = 0

        return self.state, {}

    def render(self, return_image=True):
        plt.cla()
        plt.xlim(*self.plt_xlim)
        plt.ylim(*self.plt_xlim)

        cart_width = 0.02
        cart_height = 0.02
        cart_x = float(self.state.item(0))
        cart_y = float(0)

        pendulum_1_x = float(cart_x + self.L1 * np.sin(self.state.item(1)))
        pendulum_1_y = float(cart_y + self.L1 * np.cos(self.state.item(1)))

        pendulum_2_x = float(
            pendulum_1_x + self.L2 * np.sin(self.state.item(2)))
        pendulum_2_y = float(
            pendulum_1_y + self.L2 * np.cos(self.state.item(2)))

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
