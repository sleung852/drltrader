import argparse
import pandas as pd
import os

from environ import SimStocksEnv,PortfolioEnv, PortfolioEnv2
from data import AssetData, MultiAssetData
from train import DRLAlgoTraderTrainer

import logging

if __name__ == '__main__':
    
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser()
    # environment settings
    parser.add_argument('--tickers', type=str)
    parser.add_argument("--outdir", type=str, default="result")
    args = parser.parse_args()
    
    tickers = [ticker.lower() for ticker in args.tickers.split('_')]
    
    if len(tickers) == 1:
        bm_data_dir = f'data/{tickers[0]}_finance_data_test.csv'
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
            name = f'benchmark_{tickers[0]}',
            agent = None,
            test_env = bm_env,
        )
        
        rewards, actions, ts, time = trainer.benchmark_BnH()
        
        result = pd.DataFrame(
            {
                'step': ts,
                'action': actions,
                'reward': rewards,
                'time': time
            }
        )
        result.to_csv(os.path.join(args.outdir, f'benchmark_{tickers[0]}.csv'), index=False)
            
    elif len(tickers) >= 2:
        test_data = MultiAssetData(tickers,
                                   mode='test')
        
        env_params = {
            # environment related
            'tickers': tickers,
            'mode': "test",
            'random_offset': False,
            'dim_mode': 1,
            # state related
            'commission': 0.0,
            'window_size': 5,
            'cash': False,
            'norm_func': 'linear'
        }
        
        bm_env = PortfolioEnv(test_data, env_params)
        
        trainer = DRLAlgoTraderTrainer(
            name = f'benchmark_{args.tickers}',
            agent = None,
            test_env = bm_env,
        )
        
        rewards, actions, ts, time = trainer.benchmark_BnH_multi(len(tickers), 2)
        
        
        
    result = pd.DataFrame(
        {
            'step': ts,
            'time': time,
            'action': actions,
            'reward': rewards,
            
        }
    )
    result.to_csv(os.path.join(args.outdir, f'benchmark_{args.tickers}.csv'), index=False)