import argparse
import functools

import torch
import numpy as np
import pfrl
import pandas as pd
import os
from pathlib import Path

from environ import SimStocksEnv
from model import DuellingNet, FCNet
from data import AssetData
from train import DRLAlgoTraderTrainer
from util import save_config, check_and_create_folder

import logging

if __name__ == '__main__':
    
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser()
    # environment settings
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--ticker', type=str, default='aapl')
    parser.add_argument('--shortsell', type=bool, default=False) 
    parser.add_argument('--commission', type=float, default=0.01) 
    parser.add_argument('--indicators', type=bool, default=False)
    parser.add_argument('--news', type=bool, default=False)
    parser.add_argument('--daily', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=1126) 
    # training settings
    parser.add_argument("--name", type=str, default='')
    parser.add_argument("--num_envs", type=int, default=2)
    parser.add_argument("--gpu", type=bool, default=True)
    parser.add_argument('--window_size', type=int, default=10)
    parser.add_argument('--random_offset', type=bool, default=True)
    parser.add_argument('--steps', type=int, default=5e6)
    parser.add_argument("--outdir", type=str, default="results")
    parser.add_argument("--gamma", type=float, default=0.99)
    # network settings
    parser.add_argument('--model', type=str, default='FC')
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--load_model', type=str, default='')
    args = parser.parse_args()
    
    env_params = {
            # environment related
            'mode': 'train',
            'random_offset': args.random_offset,
            'cnn': args.model == 'CNN',
            'rnn': False,
            # state related
            'commission': args.commission, 
            'window_size': args.window_size,
            'dim_mode': 2 if args.model == 'CNN' else 1,
            'shortsell': args.shortsell
        }
    
    env_params_val = env_params.copy()
    env_params_val['mode'] = 'test'
    
    ticker = args.ticker.lower()
    train_data_dir = f'data/{ticker}_finance_data_train.csv'
    val_data_dir = f'data/{ticker}_finance_data_val.csv'
    test_data_dir = f'data/{ticker}_finance_data_test.csv'
    
    # temporary solution
    if args.indicators:
        indicators = ['MACD', 'EMA', 'MA', 'RSI', 'NATR', 'OBV']
    else:
        indicators = []
    
    
    train_data = AssetData(train_data_dir,
                           daily=args.daily,
                           indicators=indicators, #args.indicators,
                           news=args.news,
                           mode='train')
    
    eval_data = AssetData(val_data_dir,
                           daily=args.daily,
                           indicators=indicators, #args.indicators,
                           news=args.news,
                           mode='eval')
    
    test_data = AssetData(test_data_dir,
                           daily=args.daily,
                           indicators=indicators, #args.indicators,
                           news=args.news,
                           mode='test')
    
    process_seeds = np.arange(args.num_envs) + args.seed * args.num_envs
    
    def make_train_env(seed, test):
        env = SimStocksEnv(train_data, env_params)
        env_seed = 2 ** 32 - 1 - seed if test else seed
        env.seed(int(env_seed))
        return env
    
    def make_batch_make_train_env(test):
        return pfrl.envs.MultiprocessVectorEnv(
            [
                functools.partial(make_train_env, process_seeds[idx], test)
                for idx, env in enumerate(range(args.num_envs))
            ]
        )
        
    def make_eval_env(seed, test):
        env = SimStocksEnv(eval_data, env_params_val)
        env_seed = 2 ** 32 - 1 - seed if test else seed
        env.seed(int(env_seed))
        return env
    
    def make_batch_make_eval_env(test):
        return pfrl.envs.MultiprocessVectorEnv(
            [
                functools.partial(make_eval_env, process_seeds[idx], test)
                for idx, env in enumerate(range(args.num_envs))
            ]
        )
    
    test_env = SimStocksEnv(test_data, env_params_val)
    obs_size = test_env.observation_space.low.size
    n_actions = test_env.action_space.n
    
    if args.model == 'Duelling':
        q_func = DuellingNet(
                obs_size,
                n_actions,
                args.hidden_size,
        )
    elif args.model == 'FC':
        q_func = FCNet(
            obs_size= obs_size,
            n_actions= n_actions,
            hidden_size=args.hidden_size
        )

    optimizer = torch.optim.Adam(
        q_func.parameters(),
        lr=1e-4,
        weight_decay=1e-5
    )
    
    replay_buffer = pfrl.replay_buffers.PrioritizedReplayBuffer(
        capacity=1e5
    )
    
    explorer = pfrl.explorers.LinearDecayEpsilonGreedy(
        start_epsilon=1.0, end_epsilon=0.1,
        decay_steps=1e6, random_action_func=test_env.action_space.sample
        )

    agent = pfrl.agents.DoubleDQN(
        q_function=q_func,
        optimizer=optimizer,
        replay_buffer=replay_buffer,
        gamma=args.gamma,
        explorer=explorer,
        minibatch_size=128,
        replay_start_size=10000,
        update_interval=100,
        target_update_interval=1000,
        gpu=torch.cuda.current_device() if args.gpu else -1,
    )
    
    if args.load_model != '':
        agent.load(args.model_dir)
        
    model_name = f'{args.model}_{args.hidden_size}_{args.window_size}_{args.ticker}'+f'_{args.name}'
    check_and_create_folder(os.path.join('result', model_name))
    save_config(env_params, os.path.join('result', model_name , 'env_params.json'))
    
    pfrl.experiments.train_agent_batch_with_evaluation(
        agent=agent,
        env=make_batch_make_train_env(test=False),
        eval_env=make_batch_make_eval_env(test=True),
        outdir=os.path.join('result', model_name),
        steps=args.steps,
        eval_n_steps=None,
        eval_interval=10000,
        eval_n_episodes=20,
        log_interval=100,
        max_episode_len=500,
        eval_max_episode_len=int(1e6),
        use_tensorboard=True
    )
    