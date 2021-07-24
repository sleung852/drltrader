import gym
import gym.spaces
from gym.utils import seeding
from gym.envs.registration import EnvSpec
from gym.utils import seeding

import pandas as pd
import numpy as np

import enum

import logging

from data import AssetData, MultiAssetData
from util import softmax, linear

"""
** Notes **
# Types of Environments                
SimStocksEnv - Handles an environment that accept Discrete actions
PortfolioStocksEnv - Handles an environment that accept Continuous actions

# Types of States
OneStockState - Handles an environment containing only one stock
                returning states as a 1D array
OneStock2DState - Handles an environment containing only one stock
                returning states as a 2D matrix
MultiStockState - Handles an environment containing multiple stocks
                returning states as a 1D array
MultiStock3DState - Handles an environment containing multiple stocks
                returning states as a 3D matrix               
"""

class SimStocksEnv(gym.Env):
    # these are required by gym.Env
    metadata = {'render.modes': ['human']}
    spec = EnvSpec("StocksEnv-v0")
    
    def __init__(self, findata, params=None):
        if params is None:
            # default parameters
            params = {
                # environment related
                'mode': 'train',
                'random_offset': True,
                'cnn': False,
                'rnn': False,
                # state related
                'commission': 0.0, 
                'n_step': 10,
                'dim_mode': 1,
                'shortsell': False
            }
        logging.info(params)
        assert isinstance(findata, AssetData)
        # checking params validity
        if params['dim_mode'] == 1:
            self.state = OneStockState(findata, params)
        elif params['dim_mode'] == 2:
            self.state = OneStock2DState(findata, params)
        else:
            raise ValueError('dim_mode must be either 1 or 2')
        if params['mode'] not in ['train', 'evaluate', 'test']:
            raise ValueError("params['mode'] must be either 'train', 'evaluate' or 'test'")
        # define spaces
        self.action_space = gym.spaces.Discrete(n=len(Actions))
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=self.state.shape, dtype=np.float32)
        # options
        self.random_offset = params['random_offset']
        
        logging.info('Environment is ready.')
    
    def reset(self):
        self.state.reset()
        return self.state.encode()
    
    def step(self, action_idx):
        action = Actions(action_idx)
        reward, done = self.state.step(action)
        obs = self.state.encode()
        info = {
            "ind": self.state.ind
        }
        return obs, reward, done, info
        
    # required by gym.Env
    def render(self, mode='human', close=False):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31
        self.state.set_seed(seed2)
        logging.info(f"Environment Seed: {seed2}")
        return [seed1, seed2]

class Actions(enum.Enum):
    Skip = 0
    Buy = 1
    Sell = 2
    
class OneStockState:
    def __init__(self, findata, params):
        self.findata = findata
        self.params = params
        self.bars_count = params['window_size']
    
    @property
    def shape(self):
        # observation per each timestep + 2 position related info
        if self.params['shortsell']:
            return (self.findata.relative_prices.shape[1]-2)*self.bars_count + 3,            
        else:
            return (self.findata.relative_prices.shape[1]-2)*self.bars_count + 2,
    
    def reset(self):
        """
        Reset environment to restart
        """
        if self.params['random_offset'] and self.params['mode'] == 'train':
            self.ind = np.random.randint(
                self.bars_count,
                self.findata.price_data.shape[0]-self.bars_count)
        else:
            self.ind = self.bars_count
        # default values
        self.trade_count = 0
        self.long_position = 0
        self.transaction_price = 0.0
        self.short_position = 0
        
    def encode(self):
        """
        Return current observation in an numpy array format (np.float32)
        """
        obs = np.zeros(shape=self.shape).astype(np.float32)
        # create price information
        counter = 0
        bar_size = self.findata.relative_prices.shape[1]-2
        for idx in range(-self.bars_count,0):
            obs[counter*bar_size:(counter+1)*bar_size] = self.findata.relative_prices.iloc[idx, 2:].values.astype(np.float32)
            counter += 1
        # create position and create position cum PnL info
        if not self.params['shortsell']:
            obs[-2] = self.long_position
            if self.long_position:
                obs[-1] = self.findata.price_data.iloc[self.ind].loc['close_1min'] / self.transaction_price - 1.0
            else:
                obs[-1] = 0.0
        else:
            obs[-3] = self.long_position
            obs[-2] = self.short_position
            if self.long_position:
                obs[-1] = self.findata.price_data.iloc[self.ind].loc['close_1min'] / self.transaction_price - 1.0
            elif self.short_position:
                obs[-1] = self.transaction_price / self.findata.price_data.iloc[self.ind].loc['close_1min'] - 1.0
            else:
                obs[-1] = 0.0            
        return obs
    
    def step(self, action):
        """
        Receive action and revert observation, reward, done, info
        """
        assert isinstance(action, Actions)
        reward = 0.0
        # check whether agent has reached the last data point
        done = (self.ind == self.findata.price_data.index[-1])
        current_price = self.findata.price_data.iloc[self.ind].loc['close_1min']
        logging.debug(self.findata.price_data.iloc[self.ind].loc['time'])
        # process actions
        if (action == Actions.Sell and self.long_position) or done:
            reward += (current_price / self.previous_price - 1.0) * 100 - self.params['commission']
            self.long_position = 0
            self.transaction_price = 0.0
            self.trade_count += 1
        elif action == Actions.Buy and not self.long_position and not self.short_position:
            reward -= self.params['commission']
            self.long_position = 1
            self.transaction_price = current_price
            self.trade_count += 1
        elif self.long_position:
            reward += (current_price / self.previous_price - 1.0) * 100
        elif not self.params['shortsell'] and self.long_position:
                reward += (current_price / self.previous_price - 1.0) * 100  
        elif self.params['shortsell']:
            if (action == Actions.Buy and self.short_position) or done:
                reward += (self.previous_price / current_price - 1.0) * 100 - self.params['commission']*2
                self.short_position = 0
                self.transaction_price = 0.0
                self.trade_count += 1
            elif action == Actions.Sell and not self.short_position and not self.long_position:
                reward -= self.params['commission']*2
                self.short_position = 1
                self.transaction_price = current_price
                self.trade_count += 1
            elif self.short_position:
                reward += (self.previous_price / current_price - 1.0) * 100  
            elif self.long_position:
                reward += (current_price / self.previous_price - 1.0) * 100  
        # elif not self.long_position and self.mode=='train':
        #     reward -= 0.0001 # penalty for idle in training
        self.previous_price = current_price
        self.ind += 1
        return reward, done
    
    def set_seed(self, seed):
        np.random.seed(seed)
    
class OneStock2DState(OneStockState):
    
    @property
    def shape(self):
        if self.params['cnn']:
            adj_shape = max((self.findata.relative_prices.shape[1]-2, self.bars_count + 1))
            return adj_shape, adj_shape
        elif self.params['rnn']:
            return self.bars_count + 1, self.findata.relative_prices.shape[1]-2
        else:
            raise ValueError("Incorrectly using 2D state: either params['cnn'] or params['rnn'] should be True")
    
    def encode(self):
        """
        Return current observation in numpy 2D matrix
        """
        obs = np.zeros(shape=self.shape).astype(np.float32)
        # create price data
        bar_depth = self.findata.relative_prices.shape[1]-2
        obs[:self.bars_count,:bar_depth] = self.findata.relative_prices.iloc[self.ind-self.bars_count:self.ind, 2:].values.astype(np.float32)
        # create position and create position PnL info
        # in last row
        obs[-1,0] = self.long_position
        if self.long_position:
            obs[-1,1] = self.findata.price_data.iloc[self.ind].loc['close_1min'] / self.transaction_price - 1.0
        else:
            obs[-1,1] = 0.0
        return obs
    
class PortfolioEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    spec = EnvSpec("StocksEnv-v1")
    
    def __init__(self, multiassetdata, params=None):
        # default parameters
        if params is None:
            params = {
                # environment related
                'tickers': ['aapl', 'amzn'],
                'mode': 'train',
                'random_offset': True,
                'cnn': False,
                'rnn': False,
                # state related
                'commission': 0.01, 
                'n_step': 10,
                'dim_mode': 1,
            }
        if params['dim_mode'] == 1:
            self.state = MultiStockState(multiassetdata, params)
        elif params['dim_mode'] == 3:
            self.state = MultiStock3DState(multiassetdata, params)
        else:
            raise ValueError('dim_mode must be either 1 or 3')
        if params['mode'] not in ['train', 'evaluate', 'test']:
            raise ValueError("params['mode'] must be either 'train', 'evaluate' or 'test'")
        
        if params['cash']:
            self.action_space = gym.spaces.Box(
                low=0, high=1,
                shape=(len(params['tickers'])+2,), dtype=np.float32
            ) # 1 is for the position for cash, 1 for postion adjustment boolean
        else:
            self.action_space = gym.spaces.Box(
                low=0, high=1,
                shape=(len(params['tickers'])+1,), dtype=np.float32
            ) # 1 for postion adjustment boolean
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=self.state.shape, dtype=np.float32
        )
        # options
        self.random_offset = params['random_offset']
    
    def reset(self):
        self.state.reset()
        return self.state.encode()
    
    def step(self, action):
        reward, done, info_s = self.state.step(action)
        obs = self.state.encode()
        info = {
            "date": info_s[4],
            "positions": info_s[0],
            "close": info_s[1],
            "previous_close": info_s[2],
            "adj_port": info_s[3]
        }
        return obs, reward, done, info
        
    # required by gym.Env
    def render(self, mode='human', close=False):
        pass

    def close(self):
        pass
        
    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31
        self.state.set_seed(seed2)
        logging.info(f"Environment Seed 1: {seed2}")
        return [seed1, seed2]
    
class MultiStockState:
    def __init__(self, findata, params):
        # settings
        self.ind = None
        self.last_adj_close_price = None
        self.findata = findata
        self.params = params
        self.tickers = params['tickers']
        self.bars_count = params['window_size']
        print(params)
    
    @property
    def shape(self):
        # every column except 'time' * n_step + info about each ticker + info about cash
        if self.params['cash']:
            return (self.findata.relative_prices.shape[1]-1)*self.bars_count + len(self.tickers) + 1,
        else:
            return (self.findata.relative_prices.shape[1]-1)*self.bars_count + len(self.tickers),
    
    def reset(self):
        # default values
        if self.params['cash']:
            self.positions = np.array([1.0/len(self.tickers)]*(len(self.tickers)+1))
        else:
            self.positions = np.array([1.0/len(self.tickers)]*len(self.tickers))
        self.positions_order = self.positions
        self.last_pos = self.positions
        self.bought_price = np.zeros(len(self.tickers)) # why?
        if self.params['random_offset']:
            self.ind = np.random.randint(0, self.findata.price_data.shape[0]-self.bars_count)
        else:
            self.ind = 0
        self.trade_count = 0
        
    def encode(self):
        """
        Return current state in a 1D numpy array
        """
        obs = np.ndarray(shape=self.shape, dtype=np.float32)
        # create price information
        counter = 0
        bar_size = self.findata.relative_prices.shape[1]-1
        for idx in range(-self.bars_count+1,1):
            obs[counter*bar_size:(counter+1)*bar_size] = self.findata.relative_prices.iloc[idx, 1:].values.astype(np.float32)
            counter += 1
        # create position and create position PnL info
        if self.params['cash']:
            obs[-len(self.tickers)-1:] = self.positions.reshape(len(self.tickers)+1,)
        else:
            obs[-len(self.tickers):] = self.positions.reshape(len(self.tickers),)
        return obs
    
    def step(self, action):
        done = self.ind == self.findata.price_data.index[-1]
        change_position_bool = action[0] > 0.5
        if self.params['norm_func'] == 'softmax':
            pos = softmax(action[1:])
        elif self.params['norm_func'] == 'linear':
            pos = linear(action[1:])
        if self.params['cash']:
            adj_close_price = np.array([1.0] + [self.findata.price_data.iloc[self.ind].loc[f'{ticker}_close_1min'] for ticker in self.tickers], dtype=np.float32)
        else:
            adj_close_price = np.array([self.findata.price_data.iloc[self.ind].loc[f'{ticker}_close_1min'] for ticker in self.tickers], dtype=np.float32)
        # initialise last_adj_close_price if not initialised
        if self.last_adj_close_price is None:
            self.last_adj_close_price = adj_close_price
        info = [pos, adj_close_price, self.last_adj_close_price, change_position_bool, self.findata.price_data.iloc[self.ind].loc['time']]           
        # if the positions_order differs from current positions,
        # this means in last timestep, agent submitted a new position order
        # fill order, assuming transaction price at previous close
        if not np.array_equal(self.positions_order, self.positions):
            reward = (np.divide(np.multiply(adj_close_price, self.positions_order), np.multiply(self.last_adj_close_price, self.positions)) - 1.0).sum() - self.params['commission']
            self.positions = self.positions_order
        else: # pnl from simply holding the positions
            reward = (np.divide(np.multiply(adj_close_price, self.positions), np.multiply(self.last_adj_close_price, self.positions)) - 1.0).sum()    
        # placing order
        if change_position_bool:
            self.new_positions = pos
        self.last_adj_close_price = adj_close_price
        self.ind += 1
        return reward, done, info

    def set_seed(self, seed):
        np.random.seed(seed)
            
class MultiStock3DState(MultiStockState):

    @property
    def shape(self):
        # n_step, every column except 'time' for each ticker, each ticker + additional info about portfolio status
        return (int((self.relative_prices.shape[1]-1)/len(self.tickers)), self.bars_count, len(self.tickers) + 1) # + len(self.tickers)*2,
        
    def encode(self):
        """
        Return current state into an 3D numpy array
        """
        obs = np.ndarray(shape=self.shape, dtype=np.float32)
        # create price information
        for i, ticker in enumerate(self.tickers):
            cols = [col.format(ticker.lower()) for col in ['{}_high_1min', '{}_low_1min', '{}_close_1min', '{}_volume_1min']]
            for day in [1,5,15,30,100]:
                for new_col in ['{}_close_{}d', '{}_volume_{}d']:
                    cols.append(new_col.format(ticker, day))
            obs[:,:,i] = self.relative_prices.iloc[self.ind-self.bars_count:self.ind, :].loc[:,cols].values.astype(np.float32)
        # create position and create position PnL info
        obs[:len(self.tickers)+1,0,-1] = self.positions.reshape(len(self.tickers)+1,)
        # obs[len(self.tickers):len(self.tickers)*2, -1] = self.current_pnl.reshape(len(self.tickers),1)
        return obs