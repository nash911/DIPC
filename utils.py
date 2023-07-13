import numpy as np
import torch
import matplotlib.pyplot as plt

from typing import Sequence


def batchify_obs(obs, device):
    # convert to torch
    obs = torch.tensor(obs).float().to(device).reshape(1, -1)

    return obs


def batchify(x, device):
    # convert to torch
    x = torch.tensor(x).to(device)

    return x


def unbatchify(x):
    x = x.cpu().numpy()

    return x


def plot_ddpg_all(axs, train_t: Sequence[int], actor_loss: Sequence[float],
                  critic_loss: Sequence[float], train_reward: Sequence[float],
                  train_episode_t: Sequence[int], eval_reward: Sequence[float],
                  eval_episode_t: Sequence[int], train_episode_len: Sequence[int],
                  eval_episode_len: Sequence[int], text: str = None, show: bool = False,
                  save: bool = False, path: str = None) -> None:
    """
       Method for plotting learning curves during policy training.

       Training loss, normalized episodic training and evaluation rewards, and
       percentage of transitions with goal = 'Right' are plotted.

       Parameters
       ----------
       axs:
           A list of subplots.
       train_t : Sequence[int]
           A list of time-steps pertaining to training-loss.
       train_loss : Sequence[float]
           A list of training-losses.
       train_reward : Sequence[float]
           A list of training rewards.
       train_episode_t : Sequence[int]
           A list of time-steps pertaining to the training rewards.
       eval_reward : Sequence[float]
           A list of evaluation rewards.
       eval_episode_t: Sequence[int]
           A list of time-steps pertaining to the evaluation rewards.

    """

    # Training Loss plot
    axs[0].clear()
    axs[0].plot(train_t, actor_loss, color='purple', label='Actor')
    axs[0].plot(train_t, critic_loss, color='black', label='Critic')
    axs[0].set(title='Training Loss')
    axs[0].set(ylabel='Loss')
    axs[0].set(xlabel='Timestep')
    axs[0].legend(loc='upper left')

    # Normalized episodic reward of the policy during training and evaluation
    axs[1].clear()
    axs[1].plot(train_episode_t, train_reward, color='red', label='Train')
    axs[1].plot(eval_episode_t, eval_reward, color='blue', label='Evaluation')
    axs[1].set(title='Normalized Episode Reward')
    axs[1].set(ylabel='Normalized Reward')
    axs[1].set(xlabel='Timestep')
    axs[1].legend(loc='upper left')

    # Training and Evaluation episode length
    axs[2].clear()
    axs[2].plot(train_episode_t, train_episode_len, label='Train-Episode Len.',
                color='red')
    axs[2].plot(eval_episode_t, eval_episode_len, color='blue', label='Eval-Episode Len.')
    axs[2].set(title="Episode Length")
    axs[2].set(ylabel='Steps')
    axs[2].set(xlabel='Timestep')
    axs[2].legend(loc='lower left')

    if text is not None:
        x_min = axs[0].get_xlim()[0]
        y_max = axs[0].get_ylim()[1]
        axs[0].text(x_min * 1.0, y_max * 1.2, text, fontsize=14, color='Black')

    if save:
        plt.savefig(path + "plots/learning_curves.png")

    if show:
        plt.show(block=False)
        plt.pause(0.01)


def plot_ppo_all(axs, train_t: Sequence[int], train_loss: Sequence[float],
                 train_reward: Sequence[float], train_episode_t: Sequence[int],
                 eval_reward: Sequence[float], eval_episode_t: Sequence[int],
                 train_episode_len: Sequence[int], eval_episode_len: Sequence[int],
                 log_std: Sequence[int] = None, text: str = None, show: bool = False,
                 save: bool = False, path: str = None) -> None:
    """
       Method for plotting learning curves during policy training.

       Training loss, normalized episodic training and evaluation rewards, and
       percentage of transitions with goal = 'Right' are plotted.

       Parameters
       ----------
       axs:
           A list of subplots.
       train_t : Sequence[int]
           A list of time-steps pertaining to training-loss.
       train_loss : Sequence[float]
           A list of training-losses.
       train_reward : Sequence[float]
           A list of training rewards.
       train_episode_t : Sequence[int]
           A list of time-steps pertaining to the training rewards.
       eval_reward : Sequence[float]
           A list of evaluation rewards.
       eval_episode_t: Sequence[int]
           A list of time-steps pertaining to the evaluation rewards.
       right_trans_perc: Sequence[float]
           A list transition percentages.

    """

    # Training Loss plot
    axs[0].clear()
    axs[0].plot(train_t, train_loss, color='red', label='Train')
    axs[0].set(title='Training Loss')
    axs[0].set(ylabel='Loss')
    axs[0].set(xlabel='Episode')
    axs[0].legend(loc='upper left')

    # Normalized episodic reward of the policy during training and evaluation
    axs[1].clear()
    axs[1].plot(train_episode_t, train_reward, color='red', label='Train')
    axs[1].plot(eval_episode_t, eval_reward, color='blue', label='Evaluation')
    axs[1].set(title='Normalized Episode Reward')
    axs[1].set(ylabel='Normalized Reward')
    axs[1].set(xlabel='Episode')
    axs[1].legend(loc='upper left')

    # Percentage of transitions in replay memory with episode goal = "Right"
    axs[2].clear()
    axs[2].plot(train_episode_t, train_episode_len, label='Train Episode Len.',
                color='red')
    axs[2].plot(eval_episode_t, eval_episode_len, color='blue', label='Eval Episode Len.')
    axs[2].set(title="Percentage of Transitions with Goal = 'Right'")
    axs[2].set(ylabel='Episode Len')
    axs[2].set(xlabel='Episode')
    axs[2].legend(loc='upper right')

    # print(f"log_std:\n{log_std}")

    if log_std is not None:
        axs[3].clear()
        axs[3].plot(train_t, log_std, color='brown', label='LogSTD')
        axs[3].set(title="Percentage of Transitions with Goal = 'Right'")
        axs[3].set(ylabel='LogSTD')
        axs[3].set(xlabel='Episode')
        axs[3].legend(loc='upper right')

    if text is not None:
        x_min = axs[0].get_xlim()[0]
        y_max = axs[0].get_ylim()[1]
        axs[0].text(x_min * 1.0, y_max * 1.2, text, fontsize=14, color='Black')

    if save:
        plt.savefig(path + "plots/learning_curves.png")

    if show:
        plt.show(block=False)
        plt.pause(0.01)
