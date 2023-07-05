import numpy as np
import random
import argparse

import torch
import torch.nn as nn

from dip_env import DoubleInvertedPendulumCartEnv
from ddpg import DDPG


def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set random generator seed
    SEED = 0
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    train_env = DoubleInvertedPendulumCartEnv()

    print(f"env.observation_space.size(): {train_env.observation_space.shape}")
    print(f"env.action_space: {train_env.action_space.shape}")

    # Create an additional environment for evaluation
    eval_env = DoubleInvertedPendulumCartEnv()

    # A third environment for final rendering
    render_env = DoubleInvertedPendulumCartEnv(render_mode='human')

    # Loss function for optimization - Mean Squared Error loss
    mse_loss = nn.MSELoss()

    # The DoubleDQL class object
    dqn = DDPG(train_env=train_env, eval_env=eval_env, render_env=render_env,
               loss_fn=mse_loss, gamma=GAMMA, rho=RHO, stdev=STDEV, actor_lr=ACTOR_LR,
               critic_lr=CRITIC_LR, replay_mem_size=REPLAY_MEM_SIZE, hl1_size=HL1_SIZE,
               hl2_size=HL2_SIZE, device=device)

    # Train the agent
    dqn.train(
        training_steps=args.max_train_steps, init_training_period=INITIAL_PERIOD,
        batch_size=BATCH_SIZE, evaluation_freq=EVALUATION_FREQUENCY, verbose=args.verbose,
        episode_len=EPISODE_LENGTH, show_plot=args.plot)


if __name__ == '__main__':
    # Adam
    # ------
    ACTOR_LR = 0.001
    CRITIC_LR = 0.001
    BETA_1 = 0.9
    BETA_2 = 0.999

    # TDL
    # -----
    GAMMA = 0.99
    RHO = 0.01
    STDEV = 0.1

    # NN
    # ----
    HL1_SIZE = 64#48
    HL2_SIZE = 64#48
    BATCH_SIZE = 32

    # DQL
    # -----
    REPLAY_MEM_SIZE = 100_000
    INITIAL_PERIOD = 3000
    EPISODE_LENGTH = 1000

    # Logging
    # ---------
    EVALUATION_FREQUENCY = 5_000

    parser = argparse.ArgumentParser(description='DDPG Training for DIPCart Task')
    parser.add_argument('--max_train_steps', type=int, default=2_000_000,
                        help='maximum number of training steps (default: 2_000_000)')
    parser.add_argument('--plot',  default=False, action='store_true',
                        help='plot learning curve (default: False)')
    parser.add_argument('--verbose',  default=False, action='store_true',
                        help='exclude velocities from the observation (default: False)')
    args = parser.parse_args()

    main(args)
