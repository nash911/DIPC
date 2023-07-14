import numpy as np
import gymnasium as gym
import torch
import json
import argparse
import random

from cartpole import DIPCEnv
from td3 import TD3


def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load learning parameters
    with open(args.path + 'learning/params.dat') as pf:
        params = json.load(pf)

    # Set random generator seed
    torch.manual_seed(params['seed'])
    np.random.seed(params['seed'])
    random.seed(params['seed'])

    env = DIPCEnv(episode_len=params['episode_len'],
                  gravity=params.get('gravity_const', -9.81),
                  action_scaler=params['action_scaler'],
                  render_mode='human')

    print(f"env.observation_space: {env.observation_space.shape}")
    print(f"env.action_space.shape: {env.action_space.shape}")

    # The DoubleDQL class object
    td3 = TD3(train_env=env, eval_env=env, render_env=env, hl1_size=params['hl1_size'],
              hl2_size=params['hl2_size'], device=device)

    td3.load_policy_network(
        model_path=args.path + 'models/' + ('latest_policy.pth' if args.latest else
                                            'best_policy.pth'))

    # Evaluate saved best agent
    _ = td3.final_evaluation(num_episodes=10, episode_len=params['episode_len'],
                             fixed_init=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TD3 Evaluation for DIPCart Task')
    parser.add_argument('--path', type=str, default='models/',
                        help='Path to the trined model (default: models/)')
    parser.add_argument('--latest',  default=False, action='store_true',
                        help='Evaluate the latest policy (default: False)')
    args = parser.parse_args()

    main(args)
