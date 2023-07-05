import gym
import time
import sys
import numpy as np
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
    def __init__(self, render_mode=None):
        self.cart_mass = 0.5
        self.pendulum_mass_1 = 0.162
        self.pendulum_mass_2 = 0.203
        self.total_mass = self.cart_mass + self.pendulum_mass_1 + self.pendulum_mass_2
        self.pendulum_length_1 = 0.170
        self.pendulum_cg_1 = 0.085
        self.pendulum_length_2 = 0.314
        self.pendulum_cg_2 = 0.157
        self.gravity = 9.81
        self.inertia_1 = (self.pendulum_mass_1 * (self.pendulum_length_1** 2))/12
        self.inertia_2 = (self.pendulum_mass_2 * (self.pendulum_length_2** 2))/12
        self.tau = 0.0395

        self.render_mode = render_mode
        # if render_mode not None:
        #     if os.path.isdir(record_path):
        #         shutil.rmtree(record_path)
        #     os.makedirs(record_path)

        self.x_goal_position = 0
        #self.kinematics_integrator = 'euler'
        self.viewer = None
        self.scale = 100  # Scale factor for converting coordinates to pixels
        # self.seed()
        self.state = None #arr2mat
        self.steps_beyond_done = None
        self.x_threshold = 1.0

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
        self.min_action = -1 # min cart force, min rope force
        self.max_action = 1  # max cart force, max rope force

        self.action_space = spaces.Box(low=self.min_action,
                                       high=self.max_action,
                                       shape=(1,),
                                       dtype=np.float32)

        # Initialize the state vector
        # [theta1_dot, theta2_dot, theta1, theta2, x, x_dot]
        #self.state = np.array([0.0, 0.0, np.pi, np.pi, 0.0, 0.0])

        # Define other necessary attributes
        self.x_goal_position = 0
        self.tau1 = 0.02# Time step for the simulation
        self.counter = 0 # Taken from fregu856


        # m0 = self.cart_mass
        # m1 = self.pendulum_mass_1
        # m2 = self.pendulum_mass_2
        # L1 = self.pendulum_length_1
        # L2 = self.pendulum_length_2


        # self.h1 = m0 + m1 + m2
        # self.h2 = (0.5 * m1 + m2 ) * L1
        # self.h3 = 0.5 * m2 * L2
        # self.h4 = (1/3 * m1 + m2) * L1**2
        # self.h5 = 0.5 * m2 * L1 * L2
        # self.h6 = 1/3 * m2 * L2**2
        # self.h7 = (0.5 * m1 + m2) * L1 * self.gravity
        # self.h8 = 0.5 * m2 * L2 * self.gravity




        # Constants for the dynamic model
        self.h1 = self.cart_mass + self.pendulum_mass_1 + self.pendulum_mass_2
        self.h2 = self.pendulum_mass_1 * self.pendulum_cg_1 + self.pendulum_mass_2 * self.pendulum_length_1
        self.h3 = self.pendulum_mass_2 * self.pendulum_cg_2
        self.h4 = self.pendulum_mass_1 * (self.pendulum_cg_1 **2 ) + self.pendulum_mass_2 * (self.pendulum_length_1 **2) + self.inertia_1
        self.h5 = self.pendulum_mass_2 * self.pendulum_cg_2 * self.pendulum_length_1
        self.h6 = self.pendulum_mass_2 * (self.pendulum_cg_2 ** 2) + self.inertia_2
        self.h7 = self.pendulum_mass_1 * self.pendulum_cg_1 * self.gravity + self.pendulum_mass_2 * self.pendulum_length_1 * self.gravity
        self.h8 = self.pendulum_mass_2 * self.pendulum_cg_2 * self.gravity

    def step(self, action):

        #assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state
        theta_dot = state.item(0)
        phi_dot = state.item(1)
        theta = state.item(2)
        phi = state.item(3)
        x = state.item(4)
        x_dot = state.item(5)

        u = action
        self.counter += 1

        def func(state, u):
            theta_dot = state.item(0)
            phi_dot = state.item(1)
            theta = state.item(2)
            phi = state.item(3)
            x = state.item(4)
            x_dot = state.item(5)
            state = np.array([[theta_dot],[phi_dot], [theta],[phi],[x],[x_dot]])
            #print("Stateshape_initial:", state.shape)

        # Constants for calculating a(Q) and B(Q)
            M11 = np.matrix([[self.h4, self.h5 * np.cos(theta - phi)], [self.h5 * np.cos(theta - phi), self.h6]])
            M12 = np.matrix([[self.h2 * np.cos(theta)], [self.h3 * np.cos(phi)]])
            C1 = np.matrix([[self.h5 * np.sin(theta - phi) * ((phi_dot) **2 )], [-self.h5 * np.sin(theta - phi) * ((theta_dot) ** 2)]])
            G1 = np.matrix([[-self.h7 * np.sin(theta)], [-self.h8 * np.sin(phi)]])

            D = np.matrix([[self.h1, self.h2 * np. cos(theta), self.h3 * np.cos(phi)],
                    [self.h2 * np.cos(theta), self.h4, self.h5 * np.cos(theta-phi)],
                    [self.h3 * np.cos(phi), self.h5 * np.cos(theta-phi), self.h6]])

            G = np.matrix([[0], [-self.h7 * np.sin(theta)], [-self.h8 * np.sin(phi)]])

            H  = np.matrix([[1],[0],[0]])

            I = np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

            O_3_3 = np.matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

            O_3_1 = np.matrix([[0], [0], [0]])

            M11_inv = np.linalg.inv(M11)
            M12_mul = np.dot(M12, (-x_dot / self.tau))
            C1_G1_sum = C1 + G1
            MCG = (M12_mul + C1_G1_sum)
            result1 = np.dot(-M11_inv, MCG ) #matrix calculation in A matrix
            M12_div = (M12 / self.tau)
            result2 = np.dot(-M11_inv, M12_div) #matrix calculation in B matrix

            #W = np.bmat([[O_3_1],[np.linalg.inv(D)*G]])

            a_Q = np.matrix([[result1[0, 0]], [result1[1, 0]], [theta_dot], [phi_dot], [x_dot], [-x_dot / self.tau]])
            #print("a_Q shape:", a_Q.shape)


            B_Q = np.matrix([[result2[0, 0]], [result2[1,0]], [0], [0], [0], [1/self.tau]])
            #print("B_Q shape:", B_Q.shape)
            #print("U: " , u)
            Q_dot = a_Q  + B_Q * u #+ W
            #print("Q_dot:", Q_dot.shape)
            return Q_dot
        # solver = ode(func)
        # solver.set_integrator("dop853") # (Runge-Kutta)
        # solver.set_f_params(u)
        # t0 = 0
        # state0 = state
        # solver.set_initial_value(state0, t0)
        # solver.integrate(self.tau1) # change the self.tau here(sampling time)
        # state=solver.y

        state_dot = func(state, u)
        state_dot_new = np.matrix(state_dot)
        #print("State_dot_new: ", state_dot_new.shape)
        # print("State_dot:", state_dot.shape)
        # reshapedstate = np.reshape(state,(1,6))
        # state1 = np.add(state_dot_new, np.multiply(state_dot_new, self.tau1))
        # print(state1)
        # print("State_transposed:",reshapedstate.shape)
        # print("State1_later:", state1.shape)


        #print("ShapeSelfState:", self.state.shape)
        self.state = (np.reshape(self.state,(6,1)) + self.tau1 * state_dot_new) #np.array([state_dot[0],state_dot[1],state_dot[2],state_dot[3],state_dot[4],state_dot[5]])
        #print("ShapeSelfState1:", self.state.shape)
        self.state = np.squeeze(self.state)
        #print("ShapeSelfState2:", self.state.shape)

        flag = self.x_goal_position

        # done =  x < -self.x_threshold \
        #         or x > self.x_threshold \
        #         or self.counter > 100000 \
        #         # or theta > 90*2*np.pi/360 \
        #         # or theta < -90*2*np.pi/360
        # done = bool(done)
        reward = 0
        alive_bonus = 10
        x_tip = x + self.pendulum_length_1 * np.sin(theta) + self.pendulum_length_2 * np.sin(phi)
        y_tip = self.pendulum_length_1 * np.cos(theta) + self.pendulum_length_2 * np.cos(phi)

        # theta_dot = np.clip(theta_dot, -10, 10)
        # phi_dot = np.clip(phi_dot, -10, 10)
        # x_dot = np.clip(x_dot, -10, 10)

        dist_penalty = (0.01 * (x_tip - flag) ** 2) + ((((y_tip -0.464) )) ** 2 ) + 0.5* (1 - np.exp(-1 * (0.5 * (0.5 ** 2 * ((x - flag) **2)))))
        velocity_penalty = (0.001 * (theta_dot ** 2)) + (0.001 * (phi_dot) ** 2) + (0.005 * (x_dot **2))

        # print(alive_bonus, dist_penalty, velocity_penalty)
        reward = alive_bonus - dist_penalty - velocity_penalty
        #cost = - (((theta)**2) + (0.1 * (theta_dot) **2 ) + ((phi)**2) + (0.1 * (phi_dot) **2 ) +   0.001 * (u ** 2) + 0.001 * (x ** 2)  )
        #cost = 10*normalize_angle(theta) + 10*normalize_angle(phi)
        #reward = cost

        #reward = alive_bonus -(dist_penalty + velocity_penalty )#-cost - 0.01 * dist_penalty - 0.001 * velocity_penalty

        #done=  bool( )
        terminated = bool(x < -self.x_threshold or x > self.x_threshold or
                          theta > 90*2*np.pi/360 or theta < -90*2*np.pi/360)
        truncated = bool(self.counter >= 1000)

        if (
            x > flag - 0.1 and x < flag + 0.1
            and x_dot > -0.1 and x_dot < 0.1
            and theta_dot > -0.05 and theta_dot < 0.05
            and np.sin(theta) > -0.05 and np.sin(theta)< 0.05
            and phi_dot > -0.05 and phi_dot < 0.05
            and np.sin(phi) > -0.05 and np.sin(phi)< 0.05
        ):
            reward += 100.0

        if(x < -self.x_threshold or x > self.x_threshold):
            reward -= 100

        if self.render_mode == 'human':
            img = self.render()
            info = {'img': img}
        else:
            info = {}

        #print out -reward for each episode(calculate)
        #print total penalties for each episode
        # subtract reward when boundaries conditions are overshooted
        # self.counter measuring ??
        return self.state, reward, terminated, truncated, info



    def reset(self, seed=None):
        # Reset the state
        self.state = np.array([
            np.random.uniform(low=0, high=0),  # theta1_dot
            np.random.uniform(low=0, high=0),  # phi_dot
            np.random.uniform(low= 0, high= 0),  # theta1
            np.random.uniform(low= 0, high= 0),  # phi
            np.random.uniform(low=0, high= 0),  # x
            np.random.uniform(low=0, high=0),  # x_dot
        ])

        self.steps_beyond_done = None
        self.counter = 0
        return self.state, {}


    def render(self, return_image=True):
        plt.cla()
        plt.xlim(-2, 2)
        plt.ylim(-1, 1)
        #plt.gca().set_aspect('equal')  # Set aspect ratio to equal


        cart_width = 0.02
        cart_height = 0.02
        #print("self.stateMatrixShape:", self.state.shape)
        cart_x = float(self.state[0,4])
        cart_y = float(0)

        pendulum_1_x = float(cart_x + self.pendulum_length_1 * np.sin(self.state[0,2]))
        pendulum_1_y = float(cart_y + self.pendulum_length_1 * np.cos(self.state[0,2]))

        pendulum_2_x = float(pendulum_1_x + self.pendulum_length_2 * np.sin(self.state[0,3]))
        pendulum_2_y = float(pendulum_1_y + self.pendulum_length_2 * np.cos(self.state[0,3]))

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
        plt.plot(rect_vertices[:, 0], rect_vertices[:, 1], 'k-')  # Connect vertices to form rectangle
        plt.plot(cart_x, cart_y, 'ko', markersize=10)  # Plot a marker at the center of the rectangle        plt.gca().add_patch(cart_rect)

        plt.plot([cart_x, pendulum_1_x], [cart_y, pendulum_1_y], 'r-')
        plt.plot([pendulum_1_x, pendulum_2_x], [pendulum_1_y, pendulum_2_y], 'b-')
        #plt.plot(cart_x, cart_y, 'ko', markersize=10)
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
