import numpy as np
import random
import argparse
import os
import json
import torch

from datetime import datetime
from shutil import rmtree

from cartpole import DIPCEnv
from ppo_continuous import PPOContinuous


# Saves the parameters above in separate file for reference
def save_learning_params(training_episodes: int):
    learning_params = dict()

    # Environment
    # -----------
    learning_params['seed'] = SEED
    learning_params['episode_len'] = EPISODE_LENGTH
    learning_params['action_scaler'] = ACTION_SCALER
    learning_params['gravity_const'] = GRAVITY_CONST
    learning_params['fixed_init'] = FIXED_INIT

    # Training
    learning_params['training_episodes'] = training_episodes
    learning_params['update_epochs'] = UPDATE_EPOCHS
    learning_params['eval_freq'] = EVAL_FREQ

    # AGENT
    learning_params['hl1_size'] = HL1_SIZE
    learning_params['hl2_size'] = HL2_SIZE
    learning_params['log_std'] = LOG_STD

    # ADAM
    learning_params['learning_rate'] = LR
    learning_params['anneal_learning_rage'] = ANNEAL_LR

    # PPO
    learning_params['entrophy_coefficient'] = ENT_COEF
    learning_params['entrophy_decay'] = ENT_DECAY
    learning_params['value_function_coefficient'] = VF_COEF
    learning_params['clip_coefficient'] = CLIP_COEF
    learning_params['gamma'] = GAMMA
    learning_params['gae_lambda'] = GAE_LAMBDA
    learning_params['batch_size'] = BATCH_SIZE
    learning_params['normalize_advantage'] = NORM_ADV
    learning_params['clip_value_loss'] = CLIP_VLOSS
    learning_params['max_grad_norm'] = MAX_GRAD_NORM
    learning_params['target_kl'] = TARGET_KL

    return learning_params


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set random generator seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # Create a timestamp directory to save model, parameter and log files
    train_path = \
        ('training/PPO/' + str(datetime.now().date()) + '_' +
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

    # A third environment for final rendering
    render_env = DIPCEnv(episode_len=EPISODE_LENGTH, action_scaler=ACTION_SCALER,
                         gravity=GRAVITY_CONST, render_mode='human')

    ppo = PPOContinuous(
        env=train_env, render_env=render_env, episode_length=EPISODE_LENGTH, lr=LR,
        hl1_size=HL1_SIZE, hl2_size=HL2_SIZE, log_std=LOG_STD, device=device)

    # Save the learning parameters for reference
    learning_params = save_learning_params(args.num_episodes)

    # Dump learning params to file
    with open(train_path + 'learning/params.dat', 'w') as jf:
        json.dump(learning_params, jf, indent=4)

    # Train the agents using SelfPlay
    ppo.train(ent_coef=ENT_COEF, ent_decay=ENT_DECAY, vf_coef=VF_COEF, gamma=GAMMA,
              clip_coef=CLIP_COEF, batch_size=BATCH_SIZE, gae_lambda=GAE_LAMBDA,
              eval_freq=EVAL_FREQ, num_episodes=args.num_episodes, anneal_lr=ANNEAL_LR,
              learning_rate=LR, update_epochs=UPDATE_EPOCHS, norm_adv=NORM_ADV,
              clip_vloss=CLIP_VLOSS, max_grad_norm=MAX_GRAD_NORM, target_kl=TARGET_KL,
              episode_len=EPISODE_LENGTH, verbose=args.verbose, show_plot=args.plot,
              path=train_path)


if __name__ == "__main__":

    # Environment
    # -----------
    SEED = 0
    EPISODE_LENGTH = 200#500
    ACTION_SCALER = 10#30.0
    GRAVITY_CONST = -9.81
    FIXED_INIT = False

    # Training
    UPDATE_EPOCHS = 4
    EVAL_FREQ = 50#20
    RECORD = True
    RECORD_PATH = 'videos/images/'

    # AGENT
    HL1_SIZE = 64
    HL2_SIZE = 64
    LOG_STD = 0#-3.29

    # ADAM
    LR = 0.0001#0.0001#7.77e-05            # Learning rate
    ANNEAL_LR = False     # For decaying learning rate over time

    # PPO
    ENT_COEF = 0.1#0.00429        # Entropy
    ENT_DECAY = 0.998     # Entropy decay
    VF_COEF = 0.1#0.19         # Value function coefficient
    CLIP_COEF = 0.1       # Clip coefficient, makes sure that agent doesn't drift too far
    GAMMA = 0.99#0.9999          # Discount factor from Belman-equation (value of future rewards)
    GAE_LAMBDA = 0.99     # Generalized Advantage Estimate
    BATCH_SIZE = 32
    NORM_ADV = False#True       # Normalized advantage estimate
    CLIP_VLOSS = True     # Value network, allows to keep reward in a certain range
    MAX_GRAD_NORM = 0.000001#5  # Same for gradient
    TARGET_KL = None      # Smoothes out the KL-Distributions, but really hard to tune

    parser = argparse.ArgumentParser(description='PPO for DIPC')
    parser.add_argument('--num_episodes', type=int, default=5_000,
                        help='maximum number of training episodes (default: 3_000)')
    parser.add_argument('--continuous',  default=False, action='store_true',
                        help='Continuous version of MountainCar (default: False)')
    parser.add_argument('--plot',  default=False, action='store_true',
                        help='plot learning curve (default: False)')
    parser.add_argument('--verbose',  default=False, action='store_true',
                        help='output training logs (default: False)')
    args = parser.parse_args()

    main(args)
