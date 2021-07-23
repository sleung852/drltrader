import argparse
import pandas as pd
import os

from environ import SimStocksEnv
from data import AssetData
from train import DRLAlgoTraderTrainer

import logging

if __name__ == '__main__':
    
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser()
    # environment settings
    parser.add_argument('--ticker', type=str)
    parser.add_argument("--outdir", type=str, default="result")
    args = parser.parse_args()
    
    bm_data_dir = f'data/{args.ticker.lower()}_finance_data_test.csv'
    bm_data = AssetData(bm_data_dir,
                            daily=False,
                            indicators=[], #args.indicators,
                            news=False,
                            mode='test')
    
    env_params = {
            # environment related
            'mode': 'test',
            'random_offset': False,
            'cnn': False,
            'rnn': False,
            # state related
            'commission': 0.01, 
            'window_size': 1,
            'dim_mode': 1,
            'shortsell': False
        }
    
    bm_env = SimStocksEnv(bm_data, env_params)
    
    trainer = DRLAlgoTraderTrainer(
        name = f'benchmark_{args.ticker.lower()}',
        agent = None,
        test_env = bm_env,
    )
    
    rewards, actions, ts = trainer.benchmark_BnH()
    result = pd.DataFrame(
        {
            'step': ts,
            'action': actions,
            'reward': rewards 
        }
    )
    result.to_csv(os.path.join(args.outdir, f'benchmark_{args.ticker.lower()}.csv'), index=False)