import argparse
import functools

import torch
import numpy as np
import pfrl
import os

from drltrader.environ import PortfolioEnv, PortfolioEnv2
from drltrader.model import *
from drltrader.data import MultiAssetData
from drltrader.util import save_config, check_and_create_folder

import logging

if __name__ == '__main__':

    logging.basicConfig(format='%(asctime)s | %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p',
                        level=logging.INFO)

    parser = argparse.ArgumentParser()
    # environment settings
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--tickers', type=str, default='aapl_amzn')
    parser.add_argument('--commission', type=float, default=0.0)
    parser.add_argument('--cash', type=bool, default=False)
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
    parser.add_argument('--steps', type=int, default=2e7)
    parser.add_argument("--outdir", type=str, default="results")
    parser.add_argument("--gamma", type=float, default=0.99)
    # network settings
    parser.add_argument('--model', type=str, default='FC', choices = ['FC', 'LSTM', 'LSTM2', 'GDPG'])
    parser.add_argument('--hidden_size', type=int, default=512) # add flexibility to model
    parser.add_argument('--load_model', type=str, default='')
    parser.add_argument('--norm_func', type=str, default='linear', choices=['linear', 'softmax'])

    args = parser.parse_args()
    
    print(args)
    
    tickers = [ticker.lower() for ticker in args.tickers.split('_')]
    
    env_params = {
        # environment related
        'tickers': tickers,
        'mode': "train",
        'random_offset': args.random_offset,
        'dim_mode': 1,
        # state related
        'commission': args.commission,
        'window_size': args.window_size,
        'cash': args.cash,
        'norm_func': args.norm_func
    }
    
    print(env_params)
    
    env_params_val = env_params.copy()
    env_params_val['mode'] = 'test'
    
    # temporary solution
    if args.indicators:
        indicators = ['MACD', 'EMA', 'MA', 'RSI', 'NATR', 'OBV']
    else:
        indicators = []    

    train_data = MultiAssetData(
        tickers,
        daily=args.daily,
        indicators=indicators,
        news=args.news,
        mode="train"
    )

    eval_data = MultiAssetData(
        tickers,
        daily=args.daily,
        indicators=indicators,
        news=args.news,
        mode="val"
    )
    
    test_data = MultiAssetData(
        tickers,
        daily=args.daily,
        indicators=indicators,
        news=args.news,
        mode="test"
    )
    
    # double check from here onwards
    process_seeds = np.arange(args.num_envs) + args.seed * args.num_envs
    
    def make_train_env(seed, test):
        env = PortfolioEnv(train_data, env_params)
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
        env = PortfolioEnv(eval_data, env_params_val)
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
    
    test_env = PortfolioEnv(test_data, env_params_val)
    obs_size = test_env.observation_space.low.size
    action_size = test_env.action_space.low.size

    feature_size = test_env.state.findata.relative_prices.shape[1] - 1
    
    if args.model == 'FC':
        q_func = SimpleActor(obs_size, action_size)
        policy = SimpleCritic(obs_size, action_size, test_env.action_space.low, test_env.action_space.high)
    elif args.model == 'LSTM':
        q_func = SimpleActor(obs_size, action_size)
        policy = LSTMCritic(obs_size, action_size, test_env.action_space.low, test_env.action_space.high)
    elif args.model == 'LSTM2':
        q_func = SimpleActor(obs_size, action_size)
        policy = LSTMCritic2(action_size,
                             args.window_size,
                             feature_size,
                             len(tickers))
    elif args.model == 'GDPG':
        q_func = GDPGActor(obs_size, action_size)
        policy = GDPGCritic(action_size,
                             args.window_size,
                             feature_size,
                             len(tickers))        

    # q_func = nn.Sequential(
    #     ConcatObsAndAction(),
    #     nn.Linear(obs_size + action_size, 400),
    #     nn.ReLU(),
    #     nn.Linear(400, 300),
    #     nn.ReLU(),
    #     nn.Linear(300, 1),
    # )
    
    # policy = nn.Sequential(
    #     nn.Linear(obs_size, 400),
    #     nn.ReLU(),
    #     nn.Linear(400, 300),
    #     nn.ReLU(),
    #     nn.Linear(300, action_size),
    #     BoundByTanh(low=test_env.action_space.low, high=test_env.action_space.high),
    #     DeterministicHead(),
    #     # nn.Softmax(3),
    # )

    optimizer_actor = torch.optim.Adam(
        q_func.parameters(),
        lr=1e-3,
        weight_decay=1e-9
    )
    
    optimizer_critic = torch.optim.Adam(
        policy.parameters(),
        lr=1e-3,
        weight_decay=1e-9
    )
    
    replay_buffer = pfrl.replay_buffers.ReplayBuffer(10 ** 6)
    
    explorer = pfrl.explorers.AdditiveGaussian(
        scale=0.5, low=test_env.action_space.low, high=test_env.action_space.high
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
        minibatch_size=64,
        burnin_action_func=burnin_action_func,
    )
    
    if args.load_model != '':
        agent.load(args.model_dir)
        
    model_name = f'{args.model}_{args.hidden_size}_{args.window_size}_{args.tickers}'+f'_{args.name}'
    check_and_create_folder(os.path.join('result', model_name))
    save_config(env_params, os.path.join('result', model_name , 'env_params.json'))
    save_config(vars(args), os.path.join('result', model_name , 'config.json'))
    
    pfrl.experiments.train_agent_batch_with_evaluation(
        agent=agent,
        env=make_batch_make_train_env(test=False),
        eval_env=make_batch_make_eval_env(test=True),
        outdir=os.path.join('result', model_name),
        steps=args.steps,
        eval_n_steps=None,
        eval_interval=100000, # recommend 100000 for efficiency
        eval_n_episodes=1,
        log_interval=100,
        max_episode_len=500,
        eval_max_episode_len=int(1e5),
        use_tensorboard=True
    )
    