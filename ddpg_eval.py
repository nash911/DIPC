import numpy as np
import gymnasium as gym
import torch
import argparse
import random

from dip_env import DoubleInvertedPendulumCartEnv
from ddpg import DDPG


def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set random generator seed
    SEED = 0
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    env = DoubleInvertedPendulumCartEnv(render_mode='human')

    print(f"env.observation_space: {env.observation_space.shape}")
    print(f"env.action_space.shape: {env.action_space.shape}")

    # The DoubleDQL class object
    ddpg = DDPG(train_env=env, eval_env=env, render_env=env, hl1_size=HL1_SIZE,
                hl2_size=HL2_SIZE, device=device)

    ddpg.load_policy_network(
        model_path=args.path+('latest_policy.pth' if args.latest else 'best_policy.pth'))

    # Evaluate saved best agent
    _ = ddpg.final_evaluation(num_episodes=10)


if __name__ == '__main__':
    # NN
    # ----
    HL1_SIZE = 64#48
    HL2_SIZE = 64#48

    parser = argparse.ArgumentParser(description='DDPG Evaluation for DIPCart Task')
    parser.add_argument('--path', type=str, default='models/',
                        help='path to the trined model (default: models/)')
    parser.add_argument('--latest',  default=False, action='store_true',
                        help='Evaluate the latest policy (default: False)')
    args = parser.parse_args()

    main(args)
