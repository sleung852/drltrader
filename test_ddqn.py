import argparse

import torch
import numpy as np
import pfrl
import pandas as pd
import os
from pathlib import Path

from environ import SimStocksEnv
from model import DRQN_CustomNet, GDQN_CustomNet
from data import AssetData
from train import DRLAlgoTraderTrainer
from util import load_config, check_and_create_folder

import logging

if __name__ == '__main__':
    
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser()
    # environment settings
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--config_file', type=str, default='')
    parser.add_argument('--env_config', type=str, default='')
    parser.add_argument("--name", type=str, default='')
    parser.add_argument("--gpu", type=bool, default=True)
    parser.add_argument("--model", type=str, default='LSTM')

    args = parser.parse_args()

    if not args.config_file and not args.env_config:
        env_params = {
            # environment related
            'mode': 'test',
            'random_offset': False,
            'cnn': False,
            'rnn': True,
            # state related
            'commission': 0.01, 
            'window_size': 10,
            'dim_mode': 2,
            'shortsell': False
        }
    else:
        config = load_config(args.config_file)   
        env_params = load_config(args.env_config)
    
    env_params['mode'] = 'test'
    
    #ticker = config.ticker.lower()
    ticker = 'aapl'
    test_data_dir = f'data/{ticker}_finance_data_test.csv'
    
    # temporary solution
    if False:
    #if env_params.indicators:
        indicators = ['MACD', 'EMA', 'MA', 'RSI', 'NATR', 'OBV']
    else:
        indicators = []
    
    test_data = AssetData(test_data_dir,
                           daily=False, #config.daily,
                           indicators=indicators, #args.indicators,
                           news=False, # config.news,
                           mode='test')
    
    test_env = SimStocksEnv(test_data, env_params)
    obs_size = test_env.observation_space.shape[1]
    n_actions = test_env.action_space.n
    
    if args.model == 'LSTM':
        q_func = DRQN_CustomNet(
                obs_size,
                n_actions,
                512,
                #args.hidden_size,
                2
        )
    elif args.model == 'GRU':
        q_func = GDQN_CustomNet(
            obs_size,
            n_actions,
            512,
            #args.hidden_size,
            2
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
        gamma=0.99,#args.gamma,
        explorer=explorer,
        minibatch_size=128,
        replay_start_size=10000,
        update_interval=100,
        target_update_interval=1000,
        gpu=torch.cuda.current_device() if args.gpu else -1,
    )
    
    agent.load(args.model_dir)
    model_name = 'LSTM_512_10_aapl__test'
    #model_name = f'{config.model}_{config.hidden_size}_{config.window_size}_{config.ticker}'+f'_{config.name}'+'_test'
    
    
    trainer = DRLAlgoTraderTrainer(
        name=model_name,
        agent=agent,
        max_steps=1,
        test_env=test_env,
        outdir='result',
    )
    
    rewards, actions, ts = trainer.test_agent_detail()
    result = pd.DataFrame(
        {
            'step': ts,
            'action': actions,
            'reward': rewards 
        }
    )
    result.to_csv(os.path.join(args.model_dir, 'test_result.csv'), index=False)