# DRLTrader

## Introduction

DRLTrader is an open source toolkit to design algorithm trading algorithms in deep reinforcement learning. This is originally designed for my MSc Capstone Research in Computer Science.

![AAPL DRL GDQN Bot] (https://raw.githubusercontent.com/sleung852/drltrader/main/img/bestpnl.png/?raw=True)

### Main Features

1. DRLTrader offers various stock environments which users can use with your chosen deep reinforcement learning libraries to train your own trade bot. Various types of stock environments include single asset environment that can output 1-D array states or 2-D matrix states. Multi-asset and portfolio management designed environment is also available.
2. DRLTrader also offers a train module which allows you to train, evaluate and test your agents. However, only PyTorch is supported.



## Installation and Requirements

DRLTrader is only tested with Python 3.8 or higher. The Python package dependencies are installed automatically when you install DRLTrader.

```bash
git clone https://github.com/sleung852/drltrader
cd drltrader
python setup.py install
```



## Usage Examples

### Example 1: Train a simple Duelling DDQN agent for AAPL

1. Download AAPL intraday data from here to /data/
2. Run the following in your terminal

```bash
import os
import pathlib as Path

import numpy as np
import pandas as pd
import torch # This example uses PyTorch
import pfrl # This example uses PFRL as the DRL library

from drltrader.environ import SimStocksEnv
from drltrader.model import DuellingNet 
from drltrader.data import AssetData
from drltrader.train import DRLAlgoTraderTrainer
from drltrader.util import save_config, check_and_create_folder

# set your variables for your environment
env_params = {
  # environment related
  'mode': 'train',
  'random_offset': True, 
  'cnn': False,
  'rnn': False,
  # state related
  'commission': 0.01, # in percentage, 0.01% in commission
  'window_size': 10,
  'dim_mode': 1, # no. of dimensions for states
  'shortsell': False # no short selling is allowed
}

env_params_val = env_params.copy()
env_params_val['mode'] = 'test'

ticker = args.ticker.lower()
train_data_dir = f'data/{ticker}_finance_data_train.csv' # or replace it with your own data
val_data_dir = f'data/{ticker}_finance_data_val.csv' # or replace it with your own data
test_data_dir = f'data/{ticker}_finance_data_test.csv' # or replace it with your own data

# Asset Data can concate all necessary data together
# reducing the overhead during training
train_data = AssetData(train_data_dir,
  daily=False,
  indicators=[], # indicators can be added in the style of TA-Lib
  news=False,
  mode='train'
)

eval_data = AssetData(val_data_dir,
  daily=False,
  indicators=[], # indicators can be added in the style of TA-Lib
  news=False,
  mode='eval'
)

test_data = AssetData(test_data_dir,
  daily=False,
  indicators=[], # indicators can be added in the style of TA-Lib
  news=False,
  mode='test'
)

train_env = SimStocksEnv(train_data, env_params)
eval_env = SimStocksEnv(eval_data, env_params_val)
test_env = SimStocksEnv(test_data, env_params_val)
obs_size = train_env.observation_space.low.size
n_actions = train_env.action_space.n

q_func = DuellingNet(
  obs_size,
  n_actions,
  512, # 512 hidden size
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
  decay_steps=1e6, random_action_func=train_env.action_space.sample
)

gpu_mode = True # toggle depending on your hardware

agent = pfrl.agents.DoubleDQN(
  q_function=q_func,
  optimizer=optimizer,
  replay_buffer=replay_buffer,
  gamma=0.99,
  explorer=explorer,
  minibatch_size=128,
  replay_start_size=10000,
  update_interval=100,
  target_update_interval=1000,
  gpu=torch.cuda.current_device() if gpu_mode else -1,
)

model_name = 'DuellingDDQN'
check_and_create_folder(os.path.join('result', model_name))
save_config(env_params, os.path.join('result', model_name , 'env_params.json'))
save_config(vars(args), os.path.join('result', model_name , 'config.json'))

# Using DRLTrader's own trainer
# The benefit of using this instead of other
# deep reinforcement learning library is that
# the tensorboard tracks key
# features needed for algo trading
trainer = DRLAlgoTraderTrainer(
  name=model_name,
  agent=agent,
  max_steps=int(5e6), # run for 5 million steps
  train_env=train_env,
  eval_env=eval_env,
  test_env=test_env,
  train_max_episode_len=500,
  eval_max_episode_len=int(1e6),
  eval_episode_interval=1000,
  eval_n_episodes=1,
  outdir='result',
)

trainer.train_agent() # training progress can be monitored via tensorboard

trainer.test_agent() # the model will be tested with the test set
```



## Example 2: Duelling DDQN with short selling

A easier way to do this is to simply use train_ddqn.py

```bash
python train_ddqn.py --ticker AAPL --shortsell True --commission 0.01 --gpu True --random_offset True --steps 5000000 --model Duelling --hidden_size 512
```



## Example 3: DDQN with GRU layers with multi-environments

When training deep reinforcement learning models, it can take very long without running muliple of environments simultaneously.  

```bash
python train_batch_drqn.py --ticker AAPL --shortsell True --commission 0.01 --gpu True --random_offset True --steps 5000000 --model GDQN --hidden_size 512
```

