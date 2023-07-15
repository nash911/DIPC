import numpy as np
import random
import argparse
import os
import json

from datetime import datetime
from shutil import rmtree

import torch
import torch.nn as nn

from cartpole import DIPCEnv
from sac import SAC


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
    learning_params['entrophy_coeff_lr'] = ENTROPHY_COEFF_LR
    learning_params['beta_1'] = BETA_1
    learning_params['beta_2'] = BETA_2

    # DDPG
    # -----
    learning_params['gamma'] = GAMMA
    learning_params['rho'] = RHO
    # learning_params[''] =
    learning_params['entrophy_coeff'] = ENTROPHY_COEFF
    learning_params['auto_tune'] = AUTO_TUNE
    learning_params['log_std_min'] = LOG_STD_MIN
    learning_params['log_std_max'] = LOG_STD_MAX
    learning_params['policy_update_freq'] = POLICY_UPDATE_FREQ
    learning_params['target_update_freq'] = TARGET_UPDATE_FREQ
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
        ('training/SAC/' + str(datetime.now().date()) + '_' +
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

    train_env = DIPCEnv(episode_len=EPISODE_LENGTH, action_scaler=ACTION_SCALER,
                        gravity=GRAVITY_CONST)

    # Create an additional environment for evaluation
    eval_env = DIPCEnv(episode_len=EPISODE_LENGTH, action_scaler=ACTION_SCALER,
                       gravity=GRAVITY_CONST)

    # A third environment for final rendering
    render_env = DIPCEnv(episode_len=EPISODE_LENGTH, action_scaler=ACTION_SCALER,
                         gravity=GRAVITY_CONST, render_mode='human')

    print(f"env.observation_space.size(): {train_env.observation_space.shape}")
    print(f"env.action_space: {train_env.action_space.shape}")

    # Loss function for optimization - Mean Squared Error loss
    mse_loss = nn.MSELoss()

    # The DoubleDQL class object
    sac = SAC(train_env=train_env, eval_env=eval_env, render_env=render_env,
              init_training_period=INITIAL_PERIOD, loss_fn=mse_loss, gamma=GAMMA, rho=RHO,
              entrophy_coeff=ENTROPHY_COEFF, auto_tune=AUTO_TUNE, action_low=-1.0,
              action_high=1.0, log_std_min=LOG_STD_MIN, log_std_max=LOG_STD_MAX,
              hl1_size=HL1_SIZE, hl2_size=HL2_SIZE, replay_mem_size=REPLAY_MEM_SIZE,
              actor_lr=ACTOR_LR, critic_lr=CRITIC_LR, entrophy_coeff_lr=ENTROPHY_COEFF_LR,
              device=device)

    # Save the learning parameters for reference
    learning_params = save_learning_params(args.max_train_steps)

    # Dump learning params to file
    with open(train_path + 'learning/params.dat', 'w') as jf:
        json.dump(learning_params, jf, indent=4)

    # Train the agent
    sac.train(
        training_steps=args.max_train_steps, batch_size=BATCH_SIZE, verbose=args.verbose,
        episode_len=EPISODE_LENGTH, evaluation_freq=EVALUATION_FREQUENCY, path=train_path,
        policy_update_freq=POLICY_UPDATE_FREQ, target_update_freq=TARGET_UPDATE_FREQ,
        show_plot=args.plot)


if __name__ == '__main__':
    # Environment
    # -----------
    SEED = 0
    EPISODE_LENGTH = 200
    ACTION_SCALER = 30.0
    GRAVITY_CONST = -9.81
    FIXED_INIT = False

    # Adam
    # ------
    ACTOR_LR = 0.001
    CRITIC_LR = 0.001
    ENTROPHY_COEFF_LR = 0.001
    BETA_1 = 0.9
    BETA_2 = 0.999

    # TDL
    # -----
    GAMMA = 0.99
    RHO = 0.01
    ENTROPHY_COEFF = 0.1
    AUTO_TUNE = True
    LOG_STD_MIN = -5
    LOG_STD_MAX = 2
    POLICY_UPDATE_FREQ = 1
    TARGET_UPDATE_FREQ = 1

    # NN
    # ----
    HL1_SIZE = 64
    HL2_SIZE = 64
    BATCH_SIZE = 32

    # DQL
    # -----
    REPLAY_MEM_SIZE = 100_000
    INITIAL_PERIOD = 3000

    # Logging
    # ---------
    EVALUATION_FREQUENCY = 5_000

    parser = argparse.ArgumentParser(description='SAC Training for MountainCar')
    parser.add_argument('--max_train_steps', type=int, default=2_000_000,
                        help='Maximum number of training steps (default: 1_500_000)')
    parser.add_argument('--plot',  default=False, action='store_true',
                        help='Plot learning curve (default: False)')
    parser.add_argument('--verbose',  default=False, action='store_true',
                        help='Print train and evaluation errors(default: False)')
    args = parser.parse_args()

    main(args)
