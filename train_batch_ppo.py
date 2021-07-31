import argparse
import functools

import torch
import numpy as np
import pfrl
import os

from environ import PortfolioEnv
from model import *
from data import MultiAssetData
from util import save_config, check_and_create_folder

import logging

if __name__ == '__main__':

    logging.basicConfig(format='%(asctime)s | %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p',
                        level=logging.INFO)

    parser = argparse.ArgumentParser()
    # environment settings
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--tickers', type=str, default='aapl_amzn')
    parser.add_argument('--commission', type=float, default=0.01)
    parser.add_argument('--cash', type=bool, default=False)
    parser.add_argument('--indicators', type=bool, default=False)
    parser.add_argument('--news', type=bool, default=False)
    parser.add_argument('--daily', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=1126) 
    # training settings
    parser.add_argument("--name", type=str, default='')
    parser.add_argument("--num_envs", type=int, default=2)
    parser.add_argument("--gpu", type=bool, default=True)
    parser.add_argument('--window_size', type=int, default=50)
    parser.add_argument('--random_offset', type=bool, default=True)
    parser.add_argument('--steps', type=int, default=5e6)
    parser.add_argument("--outdir", type=str, default="results")
    parser.add_argument("--gamma", type=float, default=0.99)
    # network settings
    parser.add_argument('--model', type=str, default='PO', choices = ['PO'])
    parser.add_argument('--hidden_size', type=int, default=512) # add flexibility to model
    parser.add_argument('--load_model', type=str, default='')
    parser.add_argument('--norm_func', type=str, default='linear', choices=['linear', 'softmax'])

    args = parser.parse_args()
    
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
    
    # Normalize observations based on their empirical mean and variance
    obs_normalizer = pfrl.nn.EmpiricalNormalization(
        test_env.observation_space.low.size, clip_threshold=5
    )
    
    policy = torch.nn.Sequential(
        nn.Linear(obs_size, 256),
        nn.Tanh(),
        nn.Linear(256, 128),
        nn.Tanh(),
        nn.Linear(128, action_size),
        pfrl.policies.GaussianHeadWithStateIndependentCovariance(
            action_size=action_size,
            var_type="diagonal",
            var_func=lambda x: torch.exp(2 * x),  # Parameterize log std
            var_param_init=0,  # log std = 0 => std = 1
        ),
    )

    vf = torch.nn.Sequential(
        nn.Linear(obs_size, 256),
        nn.Tanh(),
        nn.Linear(256, 128),
        nn.Tanh(),
        nn.Linear(128, 1),
    )

    # we use orthogonal initialization as the latest openai/baselines does.
    def ortho_init(layer, gain):
        nn.init.orthogonal_(layer.weight, gain=gain)
        nn.init.zeros_(layer.bias)

    ortho_init(policy[0], gain=1)
    ortho_init(policy[2], gain=1)
    ortho_init(policy[4], gain=1e-2)
    ortho_init(vf[0], gain=1)
    ortho_init(vf[2], gain=1)
    ortho_init(vf[4], gain=1)

    # Combine a policy and a value function into a single model
    model = pfrl.nn.Branched(policy, vf)

    opt = torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-5)

    agent = pfrl.agents.PPO(
        model,
        opt,
        obs_normalizer=obs_normalizer,
        gpu=torch.cuda.current_device() if args.gpu else -1,
        update_interval=100,
        minibatch_size=64,
        epochs=5,
        clip_eps_vf=None,
        entropy_coef=0.01,
        standardize_advantages=True,
        gamma=0.995,
        lambd=0.97,
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
    