import numpy as np
import gymnasium as gym
import time
import os
import matplotlib.pyplot as plt

from typing import Sequence, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

from utils import batchify_obs, batchify, unbatchify, plot_ppo_all


class Agent(nn.Module):
    def __init__(self, inp_size: int, num_actions: int, hl_1_size: int, hl_2_size: int,
                 log_std: float = 0):
        super().__init__()

        self.network = nn.Sequential(
            self._layer_init(nn.Linear(inp_size, hl_1_size)),
            nn.ReLU(),
            self._layer_init(nn.Linear(hl_1_size, hl_2_size)),
            nn.ReLU(),
        )

        self.actor_mean = nn.Sequential(
            self._layer_init(nn.Linear(hl_2_size, num_actions), std=0.01),
            nn.Tanh(),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, num_actions) * log_std)
        self.critic = self._layer_init(nn.Linear(hl_2_size, 1))

    def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def log_std(self):
        return self.actor_logstd.item()

    def get_value(self, x=None, hidden=None):
        if hidden is None:
            hidden = self.network(x)

        return self.critic(hidden)

    def get_action(self, x, inference=False):
        hidden = self.network(x)
        action_mean = self.actor_mean(hidden)

        if not inference:
            action_logstd = self.actor_logstd.expand_as(action_mean)
            action_std = torch.exp(action_logstd)
            probs = Normal(action_mean, action_std)
            action = probs.sample()
        else:
            action = action_mean

        return action

    def get_action_and_value(self, x, action=None, action_mask=None, inference=False):
        hidden = self.network(x)
        # logits = self.actor(hidden)

        action_mean = self.actor_mean(hidden)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        if action is None:
            if not inference:
                action = probs.sample()
            else:
                action = action_mean

        return (action, probs.log_prob(action).sum(1), probs.entropy().sum(1),
                self.critic(hidden))


class PPOContinuous():
    """
        Proximal Policy Optimization Class.

        This class contains the PPO implementation.
    """

    def __init__(self, env: gym.Env, render_env: gym.Env, lr: float = 0.001,
                 episode_length: int = 200, hl1_size: int = 48, hl2_size: int = 48,
                 log_std: float = 0, seed: int = 0, device: str = "cpu"):
        """
           Parameters
           ----------
           env : gym.Env
               CartPoleLeftRight env for training the policy.
           eval_env : gym.Env
               CartPoleLeftRight env for evaluating the policy.
           gamma : float
               Discount parameter ɣ.
        """

        self.env = env
        self.render_env = render_env
        self.seed = seed
        self.device = device

        self.state_size = env.observation_space.shape[0]
        self.num_actions = env.action_space.shape[0]
        self.episode_len = episode_length

        self.agent = Agent(
            self.state_size, self.num_actions, hl1_size, hl2_size,
            log_std).to(self.device)

        self.optimizer = optim.Adam(self.agent.parameters(), lr=lr)

    def eval_agent(self):
        self.agent.eval()

    def train_agent(self):
        self.agent.train()

    def load_agent(self, model_path: str = 'models/best_policy.pth') -> None:
        """
           Method to load actor and critic networks from saved model file.
        """

        self.agent.load_state_dict(torch.load(model_path))
        self.agent.eval()

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

        # Initialize lists to store normalized eposide rewards and episode lengths
        episode_rewards = list()
        episode_lengths = list()

        # Evaluate the current policy for num_episodes number of times
        for episode in range(num_episodes):
            observation, info = self.env.reset(seed=episode)

            terminated = truncated = False
            episode_reward = 0
            episode_length = 0

            # The state -> action -> reward, next-state loop
            while not(terminated or truncated):
                state = observation
                with torch.no_grad():
                    action = self.agent.get_action(
                        x=torch.tensor(state.reshape(1, -1),
                                       dtype=torch.float32).to(self.device),
                        inference=True).cpu().numpy()
                # print(f"Action: {action}")
                observation, reward, terminated, truncated, info = \
                    self.env.step(action)
                episode_reward += (reward/episode_len)
                episode_length += 1

            # Store the normalized reward and the length of the current episode
            # in the respective list
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

        return np.mean(episode_rewards), min(episode_lengths)

    def final_evaluation(self, num_episodes: int = 10, episode_len: int = 200
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

        episode_rewards = list()
        episode_lengths = list()
        for e in range(num_episodes):
            observation, info = self.render_env.reset(seed=e)

            terminated = truncated = False
            episode_reward = 0
            episode_length = 0

            # The state -> action -> reward, next-state loop
            while not(terminated or truncated):
                state = observation
                with torch.no_grad():
                    action = self.agent.get_action(
                        x=torch.tensor(state.reshape(1, -1),
                                       dtype=torch.float32).to(self.device),
                        inference=True).cpu().numpy()
                observation, reward, terminated, truncated, info = \
                    self.render_env.step(action)
                episode_reward += (reward/episode_len)
                episode_length += 1

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            print(f"  Reward: {np.round(episode_reward, 4)} - " +
                  f"Episode Length: {episode_length}")

        observation, info = self.render_env.reset()

        # Final polict evaluation statistics
        print("\nFinal Policy Evaluation Statistics:")
        print(f"Total number of episodes: {len(episode_rewards)}")
        print(f"Mean Normalized Episodic Reward: {np.round(np.mean(episode_rewards), 4)}")
        print("Median Normalized Episodic Reward: " +
              f"{np.round(np.median(episode_rewards), 4)}")
        print("Number of early terminated episodes: " +
              f"{np.sum(np.array(episode_lengths) < episode_len)}")

        return episode_rewards

    def train(self, ent_coef: float = 0.1, ent_decay: float = 1.0, vf_coef: float = 0.1,
              gamma: float = 0.99, clip_coef: float = 0.2, batch_size: float = 32,
              gae_lambda: float = 0.95, eval_freq: int = 1000, num_episodes: int = 1_000,
              update_epochs: int = 4, anneal_lr: bool = True, learning_rate: float = 1e-3,
              norm_adv: bool = True, verbose: bool = True, clip_vloss: bool = True,
              max_grad_norm: float = 0.5, target_kl: float = None, episode_len: int = 200,
              path: str = None, show_plot: bool = False, record_path='videos/images/'):
        """
           Method for training the policy based with DDQL algorithm.

           Parameters
           ----------
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
        fig, axs = plt.subplots(4, figsize=(10, 11), sharey=False, sharex=True)

        # # Create directory to save best evaluation policy
        # if not os.path.isdir('models'):
        #     os.makedirs('models')

        # ALGO Logic: Storage setup
        obs_buff = torch.zeros((self.episode_len, self.state_size)).to(self.device)
        actions_buff = torch.zeros((self.episode_len)).to(self.device)
        logprobs_buff = torch.zeros((self.episode_len,)).to(self.device)
        rewards_buff = torch.zeros((self.episode_len,)).to(self.device)
        dones_buff = torch.zeros((self.episode_len,)).to(self.device)
        values_buff = torch.zeros((self.episode_len+1,)).to(self.device)  # Add for T+1

        # Initialize lists for storing learning curve data
        t_list = list()
        train_loss = list()
        train_reward = list()
        train_episode_len = list()
        train_episode_t = list()
        eval_reward = list()
        eval_episode_len = list()
        eval_episode_t = list()
        los_std = list()

        entrophy_list = list()

        # For learning-rate annealing
        learning_episodes = 0

        episode_reward = 0
        episode_duration = 0
        best_eval_reward = -np.inf

        self.train_agent()

        # Rollout-Store-Optimize loop
        for episode in range(1, num_episodes + 1):
            # Reset Env at the beginning of each episode
            new_obs, info = self.env.reset(seed=episode)

            # Extract the most recent observations
            next_obs = batchify_obs(new_obs, self.device)

            # Init variables for episode loop
            end_step = self.episode_len - 1

            for step in range(0, self.episode_len):
                # ALGO LOGIC: action logic
                with torch.no_grad():
                    # Save observations to rollout buffer
                    obs_buff[step] = next_obs

                    # Get the action, logp, and value for the learning-agent
                    actions, logprobs, _, values = \
                        self.agent.get_action_and_value(x=next_obs)

                # Perform one-step of the simulation
                new_obs, reward, terminated, truncated, info = \
                    self.env.step(unbatchify(actions))

                # Batchify rewards
                rewards = batchify(reward, self.device)

                # Check if the episode has terminated or truncated
                done = terminated or truncated
                terms = batchify(terminated, self.device)

                # Store transitions in the learning-agent's buffer
                obs_buff[step] = next_obs
                actions_buff[step] = actions
                logprobs_buff[step] = logprobs
                rewards_buff[step] = rewards
                dones_buff[step] = terms
                values_buff[step] = values

                # Normalized reward
                episode_reward += (reward/episode_len)

                # Episode length for plotting
                episode_duration += 1

                # Bachify observations for the next iteration
                next_obs = batchify_obs(new_obs, self.device)

                # If end of episode (terminated or truncated)
                if done:
                    # Set end-of-episode step and exit out of the episode loop
                    end_step = step + 1
                    learning_episodes += 1

                    with torch.no_grad():
                        # Calculate and save next_values to rollout buffer
                        values = self.agent.get_value(next_obs)
                        values_buff[step+1] = values
                    break

            train_reward.append(episode_reward)
            train_episode_len.append(episode_duration)
            train_episode_t.append(episode)
            episode_reward = 0
            episode_duration = 0

            # Evaluate Policy
            if (episode - 1) % eval_freq == 0 or episode == num_episodes:
                self.eval_agent()
                mean_eval_rewards, max_eval_episodes_length = \
                    self.evaluate(num_episodes=1)
                eval_reward.append(mean_eval_rewards)
                eval_episode_len.append(max_eval_episodes_length)
                eval_episode_t.append(episode)
                self.train_agent()

                if (mean_eval_rewards > best_eval_reward):
                    best_eval_reward = mean_eval_rewards
                    saved_model_txt = "Best Model Saved @ Episode %d" % episode
                    torch.save(self.agent.state_dict(), path + 'models/best_policy.pth')

                # Save the most recent model as well
                torch.save(self.agent.state_dict(), path + 'models/latest_policy.pth')

                # Plot loss, rewards, and transition percentage
                plot_ppo_all(
                    axs, t_list, train_loss, train_reward, train_episode_t, eval_reward,
                    eval_episode_t, train_episode_len, eval_episode_len, los_std,
                    show=show_plot, text=saved_model_txt, save=True, path=path)

            # bootstrap returns if not done
            advantages = {}
            returns = {}
            with torch.no_grad():
                advantages = torch.zeros_like(rewards_buff).to(self.device)
                lastgaelam = 0
                for t in reversed(range(end_step)):
                    delta = rewards_buff[t] + (gamma * values_buff[t + 1] *
                                               (1 - dones_buff[t])) - values_buff[t]

                    advantages[t] = lastgaelam = \
                        delta + (gamma * gae_lambda * (1 - dones_buff[t]) * lastgaelam)

                # Ensure to exclude T+1 value from value buffer
                returns = advantages + values_buff[:-1]

            # Annealing the rate if instructed to do so
            if anneal_lr:
                frac = 1.0 - (learning_episodes - 1.0) / num_episodes
                lrnow = frac * learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow

            # Decay Entrophy Coefficient if instructed to do so
            if ent_decay is not None:
                ent_coef *= ent_decay
            entrophy_list.append(ent_coef)

            # Optimizing the policy and value network
            clip_fracs = []

            # Get the rollouts for the previous episode for the current agent
            b_obs = obs_buff[:end_step]
            b_logprobs = logprobs_buff[:end_step]
            b_actions = actions_buff[:end_step]
            b_advantages = advantages[:end_step]
            b_returns = returns[:end_step]
            b_values = values_buff[:end_step]

            b_inds = np.arange(len(b_obs))

            for epoch in range(update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, len(b_obs), batch_size):
                    end = start + batch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(
                        x=b_obs[mb_inds], action=b_actions.long()[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clip_fracs += [
                            ((ratio - 1.0).abs() > clip_coef).float().mean().item()
                        ]

                    mb_advantages = b_advantages[mb_inds]

                    # Normalize advantage if instructed to do so
                    if norm_adv:
                        mb_advantages = ((mb_advantages - mb_advantages.mean()) /
                                         (mb_advantages.std() + 1e-8))

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio, 1 - clip_coef, 1 + clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -clip_coef,
                            clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - (ent_coef * entropy_loss) + (v_loss * vf_coef)

                    self.optimizer.zero_grad()
                    loss.backward()

                    if max_grad_norm is not None:
                        nn.utils.clip_grad_norm_(self.agent.parameters(), max_grad_norm)

                    self.optimizer.step()

                    if (len(b_obs) - end) == 0:
                        break

                if target_kl is not None:
                    if approx_kl > target_kl:
                        break

            # Calculate Explained Variance for logging
            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = (torch.tensor(np.nan) if var_y == 0 else
                             1 - np.var(y_true - y_pred) / var_y)

            train_loss.append(loss.item())
            t_list.append(episode)
            los_std.append(self.agent.log_std())

            if verbose:
                print(f"Training episode {episode}")
                print(f"Episodic Return: {train_reward[-1]}")
                print(f"Episode Length: {end_step}")
                print("")
                print(f"Value Loss: {v_loss.item()}")
                print(f"Policy Loss: {pg_loss.item()}")
                print(f"Old Approx KL: {old_approx_kl.item()}")
                print(f"Approx KL: {approx_kl.item()}")
                print(f"Clip Fraction: {np.mean(clip_fracs)}")
                print(f"Explained Variance: {explained_var.item()}")
                print("\n-------------------------------------------\n")

        end_time = time.time()
        print("\nTraining Time: %.2f(s)" % (end_time - start_time))
        input("Completed training.\nPress Enter to start the final evaluation")

        # Load the best evaluation model
        self.agent.load_state_dict(torch.load('models/best_policy.pth'))

        self.agent.eval()
        _ = self.final_evaluation(num_episodes=10)
