import time
import pandas as pd
import logging
from talib import abstract
import argparse

class AssetData:
    def __init__(self,
                 data_origin,
                 daily=False,
                 indicators=[],
                 news=False,
                 mode='train'
                 ):
        self.data_origin = data_origin
        self.daily = daily
        self.indicators = indicators
        self.news = news
        self.mode = mode
        self.price_data = None
        self.relative_prices = None
        self._create_data()
        
    def _create_data(self):
        start = time.time()
        logging.info('Preparing price data...')
        self._read_base_data()
        self._add_daily_data()
        logging.info(str(self.price_data.shape))
        logging.info(str(self.relative_prices.shape))
        self._add_indicators()
        self.relative_prices['volume_1min'] = self.relative_prices['volume_1min'].pct_change()
        if self.news:
            pass
            
    def _read_base_data(self):
        self.price_data = pd.read_csv(self.data_origin)
        self.relative_prices = self.price_data.copy()
        for col in ['high', 'low', 'close']:
            self.relative_prices[f'{col}_1min'] = (self.relative_prices[f'{col}_1min'] - self.relative_prices['open_1min']) / self.relative_prices['open_1min']
        print('debug done')
        
    def _add_daily_data(self):
        if not self.daily:
            self.price_data = self.price_data.iloc[:,:6]
            self.relative_prices = self.relative_prices.iloc[:,:6]
        else:
            for day in [1,5,15,30,100]: # in future should provide option for users to adjust
                self.relative_prices[f'close_{day}d'] = (self.relative_prices[f'close_{day}d'] - self.relative_prices['open_1min']) / self.relative_prices['open_1min']
                self.relative_prices[f'volume_{day}d'] = (self.relative_prices[f'volume_{day}d'] - self.relative_prices['volume_1min']) / self.relative_prices['volume_1min']
            
    def _add_indicators(self):
        """
        Create indicators using Ta-LIB 
        """
        if len(self.indicators) > 0:
            df_ohlc = self.price_data.iloc[:,1:6]
            df_ohlc.columns = ['open', 'high', 'low', 'close', 'volume']
            for indicator in self.indicators:
                indicator_func = abstract.Function(indicator)
                self.relative_prices[indicator_func.output_names] = indicator_func(df_ohlc)
                for output_name in indicator_func.output_names:
                    if self.relative_prices[output_name].mean() > 1:
                        self.relative_prices[output_name] /= self.relative_prices[f'open_1min']

    def _add_news(self):
        if self.news:
            if self.mode == 'train':
                df_news = pd.read_csv('data/news_train.csv')
            elif self.mode == 'test':
                df_news = pd.read_csv('data/news_test.csv')
            df_news['time'] = pd.to_datetime(df_news['time']) 
            self.relative_prices = pd.merge(self.relative_prices,
                                            df_news,
                                            how='left',
                                            on=["time"])
            
class MultiAssetData:
    def __init__(self,
                 tickers,
                 data_dir_format='data/{}_finance_data_{}.csv',
                 daily=False,
                 indicators=[],
                 news=False,
                 mode='train'
                 ):
        assert len(tickers) > 1
        self.tickers = [ticker.lower() for ticker in tickers]
        self.data_dir_format = data_dir_format
        self.daily = daily # suggest [1,5,15,30,100]
        self.indicators = indicators
        self.news = news
        self.mode = mode
        self.price_data = None
        self.relative_prices = None
        self._create_data()
    
    # quite messy at the moment, need fixing
    def _create_data(self):

        df = pd.read_csv(self.data_dir_format.format(self.tickers[0], self.mode))
        df.columns = [df.columns[0]] + [f'{self.tickers[0].lower()}_' + c for c in df.columns[1:]]
        df['time'] = pd.to_datetime(df['time'])
        # create price info for the first ticker
        if not self.daily:
            df = df.iloc[:,:6]
        if len(self.indicators) > 0:
            df_ohlc = df.iloc[:,1:6]
            df_ohlc.columns = ['open', 'high', 'low', 'close', 'volume']
            for indicator in self.indicators:
                indicator_func = abstract.Function(indicator)
                df[indicator_func.output_names] = indicator_func(df_ohlc)
                for output_name in indicator_func.output_names:
                    if df[output_name].mean() > 1:
                        df[output_name] /= df[f'{self.tickers[0].lower()}_close_1min']
        self.price_data = df
        del df
        
        for ticker in self.tickers[1:]:
            assert isinstance(ticker, str)  
            print(f'Loading {ticker} data...')    
            df_temp = pd.read_csv(self.data_dir_format.format(ticker, self.mode))
            df_temp.columns = [df_temp.columns[0]] + [f'{ticker.lower()}_' + c for c in df_temp.columns[1:]]
            df_temp['time'] = pd.to_datetime(df_temp['time'])
       
            if not self.daily:
                df_temp = df_temp.iloc[:,:6]
            # add indicators
            if len(self.indicators) > 0:
                df_ohlc = df_temp.iloc[:,1:6]
                df_ohlc.columns = ['open', 'high', 'low', 'close', 'volume']
                for indicator in self.indicators:
                    indicator_func = abstract.Function(indicator)
                    df_temp[indicator_func.output_names] = indicator_func(df_ohlc)
                    for output_name in indicator_func.output_names:
                        if df_temp[output_name].mean() > 1:
                            df_temp[output_name] /= df_temp[f'{ticker.lower()}_close_1min']

            self.price_data = pd.merge(self.price_data,
                                        df_temp,
                                        how='left',
                                        on=["time"])
        self.price_data.fillna(method='backfill', inplace=True)
        self._convert_price_data_to_relative_prices()
        
        if self.news:
            if self.mode == 'train':
                df_news = pd.read_csv('data/news_train.csv')
            elif self.mode == 'test':
                df_news = pd.read_csv('data/news_test.csv')
            df_news['time'] = pd.to_datetime(df_news['time']) 
            self.relative_prices = pd.merge(self.relative_prices,
                                            df_news,
                                            how='left',
                                            on=["time"])

    def _convert_price_data_to_relative_prices(self):
        self.relative_prices = self.price_data.copy()
        # process prices to relative prices
        for ticker in self.tickers:
            for col in ['high', 'low', 'close']:
                self.relative_prices[f'{ticker.lower()}_{col}_1min'] = (self.relative_prices[f'{ticker.lower()}_{col}_1min'] - self.relative_prices[f'{ticker.lower()}_open_1min']) / self.relative_prices[f'{ticker.lower()}_open_1min']
            if not self.only_intra:
                for day in [1,5,15,30,100]:
                    self.relative_prices[f'{ticker.lower()}_close_{day}d'] = (self.relative_prices[f'{ticker.lower()}_close_{day}d'] - self.relative_prices[f'{ticker.lower()}_open_1min']) / self.relative_prices[f'{ticker.lower()}_open_1min']
                    self.relative_prices[f'{ticker.lower()}_volume_{day}d'] = (self.relative_prices[f'{ticker.lower()}_volume_{day}d'] - self.relative_prices[f'{ticker.lower()}_volume_1min']) / self.relative_prices[f'{ticker.lower()}_volume_1min']
            # process volume to relative volume
            self.relative_prices[f'{ticker.lower()}_volume_1min'] = self.relative_prices[f'{ticker.lower()}_volume_1min'].pct_change()
            del self.relative_prices[f'{ticker.lower()}_open_1min']
            
def create_train_test(ticker,
                      train_start='2014-01-01',train_end='2019-03-31',
                      val_start='2019-04-01',val_end='2019-07-01',
                      test_start='2019-07-02',test_end='2021-05-31'):
    df = pd.read_csv(f'data/{ticker.lower()}_finance_data_full.csv')
    df['time'] = pd.to_datetime(df['time'])
    # create training data
    start_time = pd.to_datetime(train_start).tz_localize(tz=('US/Eastern'))
    end_time = pd.to_datetime(train_end).tz_localize(tz=('US/Eastern'))
    train = df[(df['time'] > start_time) & (df['time'] < end_time)]
    train.to_csv(f'data/{ticker.lower()}_finance_data_train.csv', index=False)
    # create validation data
    start_time = pd.to_datetime(val_start).tz_localize(tz=('US/Eastern'))
    end_time = pd.to_datetime(val_end).tz_localize(tz=('US/Eastern'))
    val = df[(df['time'] > start_time) & (df['time'] < end_time)]
    val.to_csv(f'data/{ticker.lower()}_finance_data_val.csv', index=False)
    # create test data
    start_time = pd.to_datetime(test_start).tz_localize(tz=('US/Eastern'))
    end_time = pd.to_datetime(test_end).tz_localize(tz=('US/Eastern'))
    test = df[(df['time'] > start_time) & (df['time'] < end_time)]
    test.to_csv(f'data/{ticker.lower()}_finance_data_test.csv', index=False)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker")
    args = parser.parse_args()
    start = time.time()
    create_train_test(args.ticker)
    print('done in {:.2f}s'.format(time.time()-start))