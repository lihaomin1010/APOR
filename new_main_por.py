from dataclasses import dataclass
from pathlib import Path

import gym
import os
import d4rl
import sys
import numpy as np
import torch
from tqdm import trange

from por import POR
from policy import GaussianPolicy
from value_functions import TwinV
from util import return_range, set_seed, Log, sample_batch, torchify, evaluate_por
import wandb
import time


def get_env_and_dataset(env_name, max_episode_steps, normalize):
    env = gym.make(env_name)
    dataset = d4rl.qlearning_dataset(env)
    if any(s in env_name for s in ('halfcheetah', 'hopper', 'walker2d')):
        min_ret, max_ret = return_range(dataset, max_episode_steps)
        print(f'Dataset returns have range [{min_ret}, {max_ret}]')
        dataset['rewards'] /= (max_ret - min_ret)
        dataset['rewards'] *= max_episode_steps
    elif 'antmaze' in env_name:
        dataset['rewards'] -= 1.

    # dones = dataset["timeouts"]
    print("***********************************************************************")
    print(f"Normalize for the state: {normalize}")
    print("***********************************************************************")
    if normalize:
        mean = dataset['observations'].mean(0)
        std = dataset['observations'].std(0) + 1e-3
        dataset['observations'] = (dataset['observations'] - mean) / std
        dataset['next_observations'] = (dataset['next_observations'] - mean) / std
    else:
        obs_dim = dataset['observations'].shape[1]
        mean, std = np.zeros(obs_dim), np.ones(obs_dim)

    for k, v in dataset.items():
        dataset[k] = torchify(v)

    return env, dataset, mean, std


def main(args):
    wandb.init(project="new_main_por",
               name=f"{args.env_name}",
               config={
                   "env_name": args.env_name,
                   "normalize": args.normalize,
                   "tau": args.tau,
                   "alpha": args.alpha,
                   "seed": args.seed,
                   "type": args.type,
                   "value_lr": args.value_lr,
                   "policy_lr": args.policy_lr,
                   "pretrain": args.pretrain,
               })
    torch.set_num_threads(1)

    env, dataset, mean, std = get_env_and_dataset(args.env_name,
                                                  args.max_episode_steps,
                                                  args.normalize)
    obs_dim = dataset['observations'].shape[1]
    act_dim = dataset['actions'].shape[1]  # this assume continuous actions
    set_seed(args.seed, env=env)

    policy = GaussianPolicy(obs_dim + obs_dim, act_dim, hidden_dim=1024, n_hidden=2)
    goal_policy = GaussianPolicy(obs_dim, obs_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden)

    por = POR(
        vf=TwinV(obs_dim, layer_norm=args.layer_norm, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden),
        policy=policy,
        goal_policy=goal_policy,
        max_steps=args.train_steps,
        tau=args.tau,
        alpha=args.alpha,
        discount=args.discount,
        value_lr=args.value_lr,
        policy_lr=args.policy_lr,
    )

    def eval_por(step):
        eval_returns = np.array([evaluate_por(env, policy, goal_policy, mean, std) \
                                 for _ in range(args.n_eval_episodes)])
        normalized_returns = d4rl.get_normalized_score(args.env_name, eval_returns) * 100.0
        wandb.log({
            'return mean': eval_returns.mean(),
            'normalized return mean': normalized_returns.mean(),
        }, step=step)

        return normalized_returns.mean()

    # pretrain behavior goal policy if needed
    if args.pretrain:
        algo_name = f"{args.type}_tau-{args.tau}_alpha-{args.alpha}_normalize-{args.normalize}"
        for step in trange(args.pretrain_steps):
            por.por_value_update(**sample_batch(dataset, args.batch_size))
        por.save_value(f"{args.model_dir}/{args.env_name}/{algo_name}")

    # train por
    else:
        por.load_value(args.load_value_path)
        algo_name = f"{args.type}_tau-{args.tau}_alpha-{args.alpha}_normalize-{args.normalize}"
        os.makedirs(f"{args.log_dir}/{args.env_name}/{algo_name}", exist_ok=True)
        eval_log = open(f"{args.log_dir}/{args.env_name}/{algo_name}/seed-{args.seed}.txt", 'w')

        for step in trange(args.train_steps):
            por.por_value_update(**sample_batch(dataset, args.batch_size))
            por.por_policy_update(**sample_batch(dataset, args.batch_size))
            if (step + 1) % args.eval_period == 0:
                average_returns = eval_por(step)
                eval_log.write(f'{step + 1}\t{average_returns}\n')
                eval_log.flush()
        eval_log.close()
        os.makedirs(f"{args.model_dir}/{args.env_name}", exist_ok=True)
        por.save(f"{args.model_dir}/{args.env_name}/{algo_name}")


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--env_name', type=str, default="antmaze-medium-diverse-v2")
    parser.add_argument('--log_dir', type=str, default="./results/")
    parser.add_argument('--model_dir', type=str, default="./models/")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--n_hidden', type=int, default=2)
    parser.add_argument('--pretrain_steps', type=int, default=10 ** 5)
    parser.add_argument('--train_steps', type=int, default=10 ** 6)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--tau', type=float, default=0.9)
    parser.add_argument('--value_lr', type=float, default=1e-4)
    parser.add_argument('--policy_lr', type=float, default=1e-4)
    parser.add_argument('--alpha', type=float, default=10.0)
    parser.add_argument('--eval_period', type=int, default=5000)
    parser.add_argument('--n_eval_episodes', type=int, default=50)
    parser.add_argument('--max_episode_steps', type=int, default=1000)
    parser.add_argument("--normalize", action='store_true')
    parser.add_argument("--layer_norm", action='store_true')
    parser.add_argument("--type", type=str, choices=['por_r', 'por_q'], default='new_por')
    parser.add_argument("--pretrain", action='store_true')
    parser.add_argument("--load_value_path", type=str, default="")
    # parser.add_argument("--ablation_type", type=str, required=True, choices=['None', 'generlization'])
    now = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    args = parser.parse_args()

    main(args)
