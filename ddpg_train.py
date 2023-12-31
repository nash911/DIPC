import numpy as np
import random
import argparse
import os
import json

from datetime import datetime
from shutil import rmtree

import torch
import torch.nn as nn

from dip_env import DoubleInvertedPendulumCartEnv
from cartpole import DIPCEnv
from ddpg import DDPG


# Saves the parameters above in separate file for reference
def save_learning_params(max_train_steps):
    learning_params = dict()

    # Environment
    # -----------
    learning_params['seed'] = SEED
    learning_params['episode_len'] = EPISODE_LENGTH
    learning_params['action_scaler'] = ACTION_SCALER
    learning_params['gravity_const'] = GRAVITY_CONST
    learning_params['fixed_init'] = FIXED_INIT

    # Adam
    # ----
    learning_params['actor_lr'] = ACTOR_LR
    learning_params['critic_lr'] = CRITIC_LR
    learning_params['beta_1'] = BETA_1
    learning_params['beta_2'] = BETA_2

    # DDPG
    # -----
    learning_params['gamma'] = GAMMA
    learning_params['rho'] = RHO
    learning_params['stdev'] = STDEV
    learning_params['replay_mem_size'] = REPLAY_MEM_SIZE
    learning_params['initial_period'] = INITIAL_PERIOD

    # NN
    # ----
    learning_params['hl1_size'] = HL1_SIZE
    learning_params['hl2_size'] = HL2_SIZE
    learning_params['batch_size'] = BATCH_SIZE

    # Training
    # --------
    learning_params['max_train_steps'] = max_train_steps
    learning_params['eval_freq'] = EVALUATION_FREQUENCY

    return learning_params


def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set random generator seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # Create a timestamp directory to save model, parameter and log files
    train_path = \
        ('training/DDPG/' + str(datetime.now().date()) + '_' +
         str(datetime.now().hour).zfill(2) + '-' + str(datetime.now().minute).zfill(2) +
         '/')

    # Delete if a directory with the same name already exists
    if os.path.exists(train_path):
        rmtree(train_path)

    # Create empty directories for saving model, parameter and log files
    os.makedirs(train_path)

    # Create a directory for storing training model and results
    os.makedirs(train_path + 'plots')
    os.makedirs(train_path + 'learning')
    os.makedirs(train_path + 'models')

    # train_env = DoubleInvertedPendulumCartEnv(episode_len=EPISODE_LENGTH, action_scaler=ACTION_SCALER, gravity=GRAVITY_CONST)
    #
    # # Create an additional environment for evaluation
    # eval_env = DoubleInvertedPendulumCartEnv(episode_len=EPISODE_LENGTH, action_scaler=ACTION_SCALER, gravity=GRAVITY_CONST)
    #
    # # A third environment for final rendering
    # render_env = DoubleInvertedPendulumCartEnv(episode_len=EPISODE_LENGTH, action_scaler=ACTION_SCALER, gravity=GRAVITY_CONST,
    #                                            render_mode='human')

    train_env = DIPCEnv(episode_len=EPISODE_LENGTH, action_scaler=ACTION_SCALER, gravity=GRAVITY_CONST)

    # Create an additional environment for evaluation
    eval_env = DIPCEnv(episode_len=EPISODE_LENGTH, action_scaler=ACTION_SCALER, gravity=GRAVITY_CONST)

    # A third environment for final rendering
    render_env = DIPCEnv(episode_len=EPISODE_LENGTH, action_scaler=ACTION_SCALER, gravity=GRAVITY_CONST,
                         render_mode='human')

    print(f"env.observation_space.size(): {train_env.observation_space.shape}")
    print(f"env.action_space: {train_env.action_space.shape}")

    # Loss function for optimization - Mean Squared Error loss
    mse_loss = nn.MSELoss()

    # The DoubleDQL class object
    dqn = DDPG(train_env=train_env, eval_env=eval_env, render_env=render_env,
               loss_fn=mse_loss, gamma=GAMMA, rho=RHO, stdev=STDEV, actor_lr=ACTOR_LR,
               critic_lr=CRITIC_LR, replay_mem_size=REPLAY_MEM_SIZE, hl1_size=HL1_SIZE,
               hl2_size=HL2_SIZE, device=device)

    # Save the learning parameters for reference
    learning_params = save_learning_params(args.max_train_steps)

    # Dump learning params to file
    with open(train_path + 'learning/params.dat', 'w') as jf:
        json.dump(learning_params, jf, indent=4)

    # Train the agent
    dqn.train(
        training_steps=args.max_train_steps, init_training_period=INITIAL_PERIOD,
        batch_size=BATCH_SIZE, evaluation_freq=EVALUATION_FREQUENCY, verbose=args.verbose,
        episode_len=EPISODE_LENGTH, show_plot=args.plot, path=train_path)


if __name__ == '__main__':
    # Environment
    # -----------
    SEED = 0
    EPISODE_LENGTH = 200#500
    ACTION_SCALER = 30.0
    GRAVITY_CONST = -9.81
    FIXED_INIT = False

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
    HL1_SIZE = 64#128#64#48
    HL2_SIZE = 64#128#64#48
    BATCH_SIZE = 32

    # DQL
    # -----
    REPLAY_MEM_SIZE = 100_000
    INITIAL_PERIOD = 3000


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
