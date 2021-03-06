import argparse

import torch
import numpy as np
import pfrl
import pandas as pd
import os
from pathlib import Path

from environ import SimStocksEnv
from model import *
from data import AssetData
from train import DRLAlgoTraderTrainer
from util import load_config, check_and_create_folder

import logging

if __name__ == '__main__':
    
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser()
    # environment settings
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--config_dir', type=str, default='')
    parser.add_argument("--name", type=str, default='')
    parser.add_argument("--gpu", type=bool, default=True)

    args = parser.parse_args()

    if not args.config_dir:
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
        config = load_config(os.path.join(args.config_dir, 'config.json'))   
        env_params = load_config(os.path.join(args.config_dir, 'env_params.json'))
    
    env_params['mode'] = 'test'
    
    print(config)
    
    ticker = config['ticker'].lower()
    test_data_dir = f'data/{ticker}_finance_data_test.csv'
    
    # temporary solution
    if config['indicators']:
        indicators = ['MACD', 'EMA', 'MA', 'RSI', 'NATR', 'OBV']
        train =  AssetData(f'data/{ticker}_finance_data_train.csv',
                           daily=config['daily'],
                           indicators=indicators,
                           news=config['news'],
                           mode='train')
        indicators_scalars = train.get_indicators_scalers()
    else:
        indicators = []
        indicators_scalars = None
    
    test_data = AssetData(test_data_dir,
                           daily=config['daily'],
                           indicators=indicators,
                           news=config['news'],
                           mode='test',
                           indicators_scalers=indicators_scalars)
    
    test_env = SimStocksEnv(test_data, env_params)
    if config["model"] in ['LSTM', 'GRU', "LSTM2", "GRU2", 'DuellingGRU']:
        obs_size = test_env.observation_space.shape[1]
    else:
        obs_size = test_env.observation_space.low.size
    n_actions = test_env.action_space.n
    
    if config["model"] == 'LSTM':
        q_func = DRQN_CustomNet(
                obs_size,
                n_actions,
                config["hidden_size"],
                2
        )
    elif config["model"] == 'GRU':
        q_func = GDQN_CustomNet(
            obs_size,
            n_actions,
            config["hidden_size"],
            2
        )
        
    elif config["model"] == 'LSTM2':
        q_func = DRQN_CustomNet2(
            obs_size,
            n_actions,
            config["hidden_size"],
            2
        )
    
    elif config["model"] == 'Duelling':
        q_func = DuellingNet(
                obs_size,
                n_actions,
                config["hidden_size"],
        )
    elif config["model"] == 'FC':
        q_func = FCNet(
            obs_size= obs_size,
            n_actions= n_actions,
            hidden_size=config["hidden_size"]
        )
    elif config["model"] == 'DRQN':
        q_func = DRQN(
            obs_size= obs_size,
            n_actions= n_actions
        )
    elif config["model"] == 'GRU2':
        q_func = GDQN_CustomNet2(
            obs_size,
            n_actions,
            config["hidden_size"],
            2
        )  
        
    elif config["model"] == 'DuellingGRU':
        q_func = DuellingGRU(
                obs_size,
                n_actions,
                config["hidden_size"]
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
        gamma=config["gamma"],
        explorer=explorer,
        minibatch_size=128,
        replay_start_size=10000,
        update_interval=100,
        target_update_interval=1000,
        gpu=torch.cuda.current_device() if args.gpu else -1,
    )
    
    agent.load(args.model_dir)
    model_name = f'{config["model"]}_{config["hidden_size"]}_{config["window_size"]}_{config["ticker"]}'+f'_{config["name"]}'+'_test'
    
    
    trainer = DRLAlgoTraderTrainer(
        name=model_name,
        agent=agent,
        max_steps=1,
        test_env=test_env,
        outdir='result',
    )
    
    rewards, actions, ts, infos = trainer.test_agent_detail()
    data = {
            'step': ts,
            'action': actions,
            'reward': rewards 
        }
    for key in infos.keys():
        data[key] = infos[key]
    result = pd.DataFrame(data)
    result.to_csv(os.path.join(args.model_dir, 'test_result.csv'), index=False)