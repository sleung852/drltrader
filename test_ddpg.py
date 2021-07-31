import argparse
import os

import torch
import numpy as np
import pandas as pd
import pfrl

from drltrader.environ import PortfolioEnv
from drltrader.model import *
from drltrader.data import MultiAssetData
from drltrader.train import DRLAlgoTraderTrainer
from drltrader.util import load_config

import logging

if __name__ == '__main__':

    logging.basicConfig(format='%(asctime)s | %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p',
                        level=logging.INFO)



    parser = argparse.ArgumentParser()

    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--config_dir', type=str, default='')
    parser.add_argument("--name", type=str, default='')
    parser.add_argument("--gpu", type=bool, default=True)

    args = parser.parse_args()

    config = load_config(os.path.join(args.config_dir, 'config.json'))   
    env_params = load_config(os.path.join(args.config_dir, 'env_params.json'))
    
    tickers = [ticker.lower() for ticker in config['tickers'].split('_')]
    
    env_params['mode'] = 'test'
    
    # temporary solution
    if config['indicators']:
        indicators = ['MACD', 'EMA', 'MA', 'RSI', 'NATR', 'OBV']
    else:
        indicators = []    

    test_data = MultiAssetData(
        tickers,
        daily=config['daily'],
        indicators=indicators,
        news=config['news'],
        mode="test"
    )
    
    test_env = PortfolioEnv(test_data, env_params)
    obs_size = test_env.observation_space.low.size
    action_size = test_env.action_space.low.size

    feature_size = test_env.state.findata.relative_prices.shape[1] - 1
    
    if config['model'] == 'FC':
        q_func = SimpleActor(obs_size, action_size)
        policy = SimpleCritic(obs_size, action_size, test_env.action_space.low, test_env.action_space.high)
    elif config['model'] == 'LSTM':
        q_func = SimpleActor(obs_size, action_size)
        policy = LSTMCritic(obs_size, action_size, test_env.action_space.low, test_env.action_space.high)
    elif config['model'] == 'LSTM2':
        q_func = SimpleActor(obs_size, action_size)
        policy = LSTMCritic2(action_size,
                             config['window_size'],
                             feature_size,
                             len(tickers))
    elif config['model'] == 'GDPG':
        q_func = GDPGActor(obs_size, action_size)
        policy = GDPGCritic(action_size,
                             config['window_size'],
                             feature_size,
                             len(tickers))        

    optimizer_actor = torch.optim.Adam(
        q_func.parameters(),
        lr=3e-5,
        weight_decay=1e-9
    )
    
    optimizer_critic = torch.optim.Adam(
        policy.parameters(),
        lr=3e-5,
        weight_decay=1e-9
    )
    
    replay_buffer = pfrl.replay_buffers.ReplayBuffer(10 ** 6)
    
    explorer = pfrl.explorers.AdditiveGaussian(
        scale=0.1, low=test_env.action_space.low, high=test_env.action_space.high
    )
    
    def burnin_action_func():
        """Select random actions until model is updated one or more times."""
        return np.random.uniform(test_env.action_space.low, test_env.action_space.high).astype(np.float32)

    agent = pfrl.agents.DDPG(
        policy,
        q_func,
        optimizer_actor,
        optimizer_critic,
        replay_buffer,
        gamma=0.99,
        explorer=explorer,
        replay_start_size=10000,
        target_update_method="soft",
        target_update_interval=1000,
        update_interval=100,
        soft_update_tau=5e-3,
        n_times_update=1,
        gpu=torch.cuda.current_device() if args.gpu else -1,
        minibatch_size=128,
        burnin_action_func=burnin_action_func,
    )
    
    agent.load(args.model_dir)
        
    #model_name = f'{config["model"]}_{args.hidden_size}_{args.window_size}_{args.tickers}'+f'_{args.name}'
    
    trainer = DRLAlgoTraderTrainer(
        name='abc',#model_name,
        agent=agent,
        max_steps=1,
        test_env=test_env,
        outdir='result',
    )
    
    rewards, actions, ts, info = trainer.test_agent_detail()
    data = {
            'step': ts,
            'action': actions,
            'reward': rewards 
        }
    
    for key in info.keys():
        data[key] = info[key]
    result = pd.DataFrame(
        data
    )
    result.to_csv(os.path.join(args.model_dir, 'test_result.csv'), index=False)
    