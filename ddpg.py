import numpy as np
import gymnasium as gym
import time
import shutil
import os
import matplotlib.pyplot as plt

from typing import Sequence, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

from utils import plot_ddpg_all


class ReplayMemory():
    """
        A class for storing and sampling transitions for Experience Replay.

        This class creates a memory buffer of a predetermined size for storing and
        sampling batch-sized transitions {sₜ, aₜ, rₜ, sₜ₊₁} during training.
    """

    def __init__(self, mem_size: int, state_size: int, action_size: int, device):
        """
           The init() method creates and initializes memory buffers for storing the
           transitions. It also initializes counters for indexing the array for
           roll-over transition insertion and sampling.

           Parameters
           ----------
           mem_size : int
               The maximum size of the replay memory buffer.
           state_size : int
               The feature size of the observations.
        """

        self.mem_size = mem_size
        self.device = device
        self.mem_count = 0
        self.current_index = 0

        self.states = np.zeros((mem_size, state_size), dtype='f8')
        self.actions = np.zeros((mem_size, action_size), dtype='f8')
        self.rewards = np.zeros((mem_size,), dtype='f8')
        self.terminals = np.zeros((mem_size,), dtype='?')

    def add(self, state: np.array, action: int, reward: float, terminal: bool) -> None:
        """
           Method for inserting a transition {sₜ, aₜ, rₜ, sₜ₊₁} to the
           replay memory buffer.

           Parameters
           ----------
           state : np.array
               An array of obsertations from the environment.
           action : int
               The action taken by the agent.
           reward : float
               The observed reward.
           terminal : bool
                A boolean indicating if state sₜ is a terminal state or not.
        """

        self.states[self.current_index % self.mem_size] = state
        self.actions[self.current_index % self.mem_size] = action
        self.rewards[self.current_index % self.mem_size] = reward
        self.terminals[self.current_index % self.mem_size] = terminal

        self.current_index = (self.current_index + 1) % self.mem_size
        self.mem_count = max(self.mem_count, self.current_index)

    def sample_batch(self, batch_size: int = 32) -> Sequence[np.array]:
        """
           Method for randomly sampling transitions {s, a, r, s'} of batch_size from
           the replay memory buffer.

           Parameters
           ----------
           batch_size : int
               Number of transitions to be sampled.

           Returns
           -------
           Sequence[np.array]
               A list of arrays each containing the sampled states, actions, rewards,
               terminal booleans, and the respective next-states (s').
        """

        while True:
            sampled_idx = \
                np.random.choice(self.mem_count, size=batch_size, replace=False)

            if (sampled_idx == self.current_index - 1).any():
                # Resample if any sampled transition is the most recently recorded
                # transition
                continue
            break

        return (
            torch.tensor(self.states[sampled_idx], dtype=torch.float32).to(self.device),
            torch.tensor(self.actions[sampled_idx], dtype=torch.int64).to(self.device),
            torch.tensor(self.rewards[sampled_idx], dtype=torch.float32).to(self.device),
            torch.tensor(self.terminals[sampled_idx], dtype=torch.int).to(self.device),
            torch.tensor(self.states[(sampled_idx + 1) % self.mem_count],
                         dtype=torch.float32).to(self.device))  # s'


class Actor(nn.Module):

    def __init__(self, stateDim: int, actionDim: int, hl1_size: int, hl2_size: int):
        super().__init__()

        # Hidden Layer 1
        self.network = nn.Sequential(
            self.init_layer(nn.Linear(in_features=stateDim, out_features=hl1_size)),
            nn.ReLU(True),
            self.init_layer(nn.Linear(in_features=hl1_size, out_features=hl2_size)),
            nn.ReLU(True),
            nn.Linear(in_features=hl2_size, out_features=actionDim),
            nn.Tanh(),)

    def init_layer(self, layer, bias_const=0.0):
        torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def forward(self, state):
        """Forward Pass"""

        return self.network(state)


class Critic(nn.Module):

    def __init__(self, stateDim: int, actionDim: int, hl1_size: int, hl2_size: int):
        super().__init__()

        # Hidden Layer 1
        self.network = nn.Sequential(
            self.init_layer(nn.Linear(in_features=stateDim+actionDim,
                                      out_features=hl1_size)),
            nn.ReLU(True),
            self.init_layer(nn.Linear(in_features=hl1_size, out_features=hl2_size)),
            nn.ReLU(True),
            nn.Linear(in_features=hl2_size, out_features=1))

    def init_layer(self, layer, bias_const=0.0):
        torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def forward(self, state, action):
        """Forward Pass"""

        return self.network(torch.concat((state, action), dim=1))


class DDPG():
    """
        Deep Deterministic Policy Gradient Class.

        This class contains the DDPG implementation.
    """

    def __init__(self, train_env: gym.Env, eval_env: gym.Env, render_env: gym.Env,
                 loss_fn=None, gamma: float = 0.99, rho: float = 0.01,
                 stdev: float = 0.1, actor_lr: float = 0.001, critic_lr: float = 0.001,
                 hl1_size: int = 48, hl2_size: int = 48, replay_mem_size: int = 100_000,
                 device: str = "cpu"):
        """
           Parameters
           ----------
           train_env : gym.Env
               Gym environment for training the policy.
           render_env : gym.Env
               Gym environment for evaluating the policy.
           loss_fn : Loss
               Loss function object for training the main-dqn.
           gamma : float
               Discount parameter ɣ.
           epsilon : float
               Exploration parameter Ɛ.
        """

        self.train_env = train_env
        self.eval_env = eval_env
        self.render_env = render_env
        self.device = device

        self.state_size = train_env.observation_space.shape[0]
        self.num_actions = train_env.action_space.shape[0]

        # Create the main Actor and Critic networks
        self.actor = Actor(
            self.state_size, self.num_actions, hl1_size, hl2_size).to(self.device)
        self.critic = Critic(
            self.state_size, self.num_actions, hl1_size, hl2_size).to(self.device)

        # Create the target Actor and Critic networks
        self.target_actor = Actor(
            self.state_size, self.num_actions, hl1_size, hl2_size).to(self.device)
        self.target_critic = Critic(
            self.state_size, self.num_actions, hl1_size, hl2_size).to(self.device)

        # Update target networks to the respective main metwork
        self.update_target_networks(rho=None)

        self.replay_memory = ReplayMemory(
            mem_size=replay_mem_size, device=device, state_size=self.state_size,
            action_size=self.num_actions)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.loss_fn = loss_fn

        self.gamma = gamma
        self.rho = rho

        self.exploration_noise = Normal(
            torch.tensor([0] * self.num_actions, dtype=float),
            torch.tensor([stdev] * self.num_actions, dtype=float))

    def update_target_networks(self, rho: float = None) -> None:
        """
           Method for updating target-actor and traget-critic parameters to that of their
           respective main networks using polyak update
        """

        if rho is None:
            # Set target networks parameters to their respective main network parameters
            self.target_actor.load_state_dict(self.actor.state_dict())
            self.target_critic.load_state_dict(self.critic.state_dict())
        else:
            # Polyak update target networks
            for target_param, param in zip(self.target_critic.parameters(),
                                           self.critic.parameters()):
                target_param.data.copy_(param.data * self.rho +
                                        target_param.data * (1.0 - self.rho))

            for target_param, param in zip(self.target_actor.parameters(),
                                           self.actor.parameters()):
                target_param.data.copy_(param.data * self.rho +
                                        target_param.data * (1.0 - self.rho))

    def load_policy_network(self, model_path) -> None:
        """
           Method to load the actor from saved model file.
        """

        self.actor.load_state_dict(torch.load(model_path))
        self.actor.eval()

    def get_action(self, state, inference: bool = False):
        action = self.actor(state)

        if not inference:
            action += self.exploration_noise.sample()

        return action.cpu().numpy()

    def evaluate(self, num_episodes: int = 3, episode_len: int = 200
                 ) -> Sequence[Union[int, float]]:
        """
           Method for evaluating policy during training.

           This method evaluates the current policy until the the min_episode number
           of episodes have been evaluated for each of the goals (Left/Right), or
           until the normalized episode reward falls below the reward_threshold.

           Parameters
           ----------
           reward_threshold : float
               The minimum reward threshold to evaluate until.
           min_episodes : int
               Minimum number of episodes to be observed for each goal type (Left/Right).

           Returns
           -------
           float
               Smallest normalized episodic reward among all evaluated episodes.
            int
               Lowest episode length among all evaluated episodes.

        """

        # Initialize lists to store normalized eposide rewards and episode length
        episode_rewards = list()
        episode_lengths = list()

        # Evaluate the current policy num_episodes number of times
        for episode in range(num_episodes):
            observation, info = self.eval_env.reset(seed=episode)

            terminated = truncated = False
            episode_reward = 0
            episode_length = 0

            # The state -> action -> reward, next-state loop
            while not(terminated or truncated):
                state = observation
                with torch.no_grad():
                    action = self.get_action(
                        torch.tensor(state, dtype=torch.float32).to(self.device),
                        inference=True)
                observation, reward, terminated, truncated, info = \
                    self.eval_env.step(action)
                episode_reward += (reward/episode_len)
                episode_length += 1

            # Store the cumulative reward and the length of the current episode
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

        # Reset the evaluation environment before exiting evaluation
        observation, info = self.eval_env.reset()

        return min(episode_rewards), max(episode_lengths)

    def final_evaluation(self, num_episodes: int = 10, record: bool = False,
                         episode_len: int = 200, record_path: str = 'videos/images/'
                         ) -> Sequence[float]:
        """
           Method for evaluating the policy at the end of training.

           Parameters
           ----------
           num_episodes : int
               Number of episodes to evaluate
           record_path : str
               Path to save rendered frame images.

           Returns
           -------
           Sequence[float]
               A list of normalized episode rewards of the evaluation
        """

        # Create or empty the current contents of the directory pointed by record_path
        if record:
            if os.path.isdir(record_path):
                shutil.rmtree(record_path)
            os.makedirs(record_path)
            n = 1

        episode_rewards = list()
        episode_lengths = list()
        for e in range(num_episodes):
            observation, info = self.render_env.reset(seed=e)

            # # Save the rendered frame as a .png image
            # if record:
            #     img = info.get('img', None)
            #     img.save(record_path + ("img_%05d.png" % n))
            #     n += 1

            terminated = truncated = False
            episode_reward = 0
            episode_length = 0

            # The state -> action -> reward, next-state loop
            while not(terminated or truncated):
                state = observation
                with torch.no_grad():
                    action = self.get_action(
                        torch.tensor(state, dtype=torch.float32).to(self.device),
                        inference=True)
                observation, reward, terminated, truncated, info = \
                    self.render_env.step(action)
                episode_reward += (reward/episode_len)
                episode_length += 1

                # Save the rendered frame as a .png image
                if record:
                    img = info.get('img', None)
                    img.save(record_path + ("img_%05d.png" % n))
                    n += 1

            # Store the cumulative reward and the length of the current episode
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            print(f"  Reward: {np.round(episode_reward, 4)} - " +
                  f"Episode Length: {episode_length}")

        # Final polict evaluation statistics
        print("\nFinal Policy Evaluation Statistics:")
        print(f"Total number of episodes: {len(episode_rewards)}")
        print(f"Mean Normalized Episodic Reward: {np.round(np.mean(episode_rewards), 4)}")
        print("Median Normalized Episodic Reward: " +
              f"{np.round(np.median(episode_rewards), 4)}")
        print("Number of early terminated episodes: " +
              f"{np.sum(np.array(episode_lengths) < episode_len)}")

        return episode_rewards

    def train(self, training_steps: int, init_training_period: int, batch_size: int = 32,
              evaluation_freq: int = 5_000, verbose: bool = True, episode_len: int = 200,
              show_plot: bool = False, path: str = None):
        """
           Method for training the policy based with DDQL algorithm.

           Parameters
           ----------
           reward_threshold : float
               Early termination reward threshold.
           training_steps : int
               Maximum number of time steps to train the policy for.
           init_training_period : int
               Number of time steps of recording transitions, before initiating
               policy training.
           main_update_period : int
               Number of time steps between consecutive main-dqn batch updates.
           target_update_period : int
               Number of time steps between consecutive target-dqn updates to main-dqn.
           batch_size : int
               Batch size for main-dqn training.
           evaluation_freq : int
               Number of time steps between policy evaluations during training.
           record_path : str
               Path to save rendered frame images.
        """

        start_time = time.time()

        # Create a matlibplot canvas for plotting learning curves
        fig, axs = plt.subplots(3, figsize=(10, 11), sharey=False, sharex=True)

        # # Create directory to save best evaluation policy
        # if not os.path.isdir('models'):
        #     os.makedirs('models')

        # Initialize lists for storing learning curve data
        t_list = list()
        critic_losses = list()
        actor_losses = list()
        train_reward = list()
        train_episode_len = list()
        train_episode_t = list()
        eval_reward = list()
        eval_episode_t = list()
        eval_episode_len = list()

        episode_reward = 0
        episode_duration = 0
        episode_count = 0
        best_eval_reward = -np.inf
        saved_model_txt = None

        self.actor.train()
        self.critic.train()

        # The state -> action -> reward, next-state loop for policy training
        observation, info = self.train_env.reset(seed=episode_count)
        for t in range(training_steps):
            state = observation

            with torch.no_grad():
                # From μ(sₜ|θ) get aₜ with exploration noise added
                action = self.get_action(
                    torch.tensor(state, dtype=torch.float32).to(self.device))

            # Step through the enviroment with action aₜ, receiving reward rₜ, and
            # observing the new state sₜ₊₁
            observation, reward, terminated, truncated, info = \
                self.train_env.step(action)

            # Save the transition {sₜ, aₜ, rₜ, sₜ₊₁} to the Replay Memory
            self.replay_memory.add(state, action, reward, (terminated or truncated))

            # Normalized reward
            episode_reward += (reward/episode_len)

            # Episode length for plotting
            episode_duration += 1

            if t > init_training_period:
                # From Replay Memory Buffer, uniformly sample a batch of transitions
                states, actions, rewards, terminals, state_primes = \
                    self.replay_memory.sample_batch(batch_size=batch_size)

                # Update Critic Network
                with torch.no_grad():
                    # Best next action estimate of the main-dqn, for the sampled batch
                    # a' = μ'(sₜ₊₁|θ')
                    action_primes = self.target_actor(state_primes)

                    # Q'(sₜ₊₁,aⱼ|θ')
                    q_primes = self.target_critic(state_primes, action_primes).reshape(-1)

                    # Target q value for the sampled batch:
                    # yⱼ = rⱼ, if sⱼ' is a terminal-state
                    # yⱼ = rⱼ + ɣ Q(sⱼ',aⱼ|θ'), otherwise.
                    target_q = rewards + (self.gamma * q_primes * (1 - terminals))

                # Predicted q value of the main-dqn, for the sampled batch
                # Q(sⱼ,aⱼ|θ)
                pred_q = self.critic(states, actions).reshape(-1)

                # Calculate loss:
                # L(θ) = 𝔼[(Q(s,a|θ) - y)²]
                critic_loss = self.loss_fn(pred_q, target_q)

                # Calculate the gradient of the critic loss w.r.t critic parameters θ
                self.critic_optimizer.zero_grad()
                critic_loss.backward()

                # Update critic network parameters θ:
                self.critic_optimizer.step()

                # Update Actor Network
                actor_loss = -self.critic(states, self.actor(states)).mean()

                # Calculate the gradient of the actor loss w.r.t actor parameters θ
                self.actor_optimizer.zero_grad()
                actor_loss.backward()

                # Update actor network parameters θ:
                self.actor_optimizer.step()

                # For plotting
                t_list.append(t)
                critic_losses.append(critic_loss.detach().numpy())
                actor_losses.append(actor_loss.detach().numpy())

                # Update target networks
                self.update_target_networks(self.rho)

            if (t + 1) % np.abs(evaluation_freq) == 0:
                # Evaluate the current policy
                self.actor.eval()
                min_eval_rewards, max_eval_episodes_length = \
                    self.evaluate(episode_len=episode_len)
                eval_reward.append(min_eval_rewards)
                eval_episode_len.append(max_eval_episodes_length)
                eval_episode_t.append(t)

                if verbose:
                    print(f"{t+1} - Train: {np.round(train_reward[-1], 4)} - " +
                          f"Evaluation: {np.round(eval_reward[-1], 4)}")

                # Save a snapshot of the best policy (main-dqn) based on the
                # evaluation results
                if (min_eval_rewards > best_eval_reward):
                    torch.save(self.actor.state_dict(), path + 'models/best_policy.pth')
                    best_eval_reward = min_eval_rewards
                    saved_model_txt = f"Best Model Saved @ Timestep {t+1} with " + \
                        f"eval reward: {np.round(best_eval_reward, 4)}"

                # Save the most recent model as well
                torch.save(self.actor.state_dict(), path + 'models/latest_policy.pth')

                # Plot loss, rewards, and transition percentage
                plot_ddpg_all(
                    axs, t_list, actor_losses, critic_losses, train_reward,
                    train_episode_t, eval_reward, eval_episode_t, show=show_plot,
                    save=True, text=saved_model_txt, train_episode_len=train_episode_len,
                    eval_episode_len=eval_episode_len, path=path)

                self.actor.train()

            # Reset the environment and store normalized episode reward on completion or
            # termination of the current episode
            if terminated or truncated:
                episode_count += 1
                observation, info = self.train_env.reset(seed=episode_count)
                train_reward.append(episode_reward)
                train_episode_len.append(episode_duration)
                train_episode_t.append(t)
                episode_reward = 0
                episode_duration = 0

        end_time = time.time()
        print("\nTraining Time: %.2f(s)" % (end_time - start_time))
        input("Completed training.\nPress Enter to start the final evaluation")

        self.actor.load_state_dict(torch.load(path + 'models/best_policy.pth'))

        self.actor.eval()
        _ = self.final_evaluation(num_episodes=10)
