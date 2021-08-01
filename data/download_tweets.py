import pandas as pd
import itertools
from datetime import timedelta, date

from snscrape.modules import twitter
import os
import time
from tqdm.notebook import tqdm, trange

import argparse

tar_acs = {
    'ftmarkets': '@FTMarkets',
    'bloombergmarkets': '@markets',
    'wsjbusiness': '@WSJbusiness',
    'reutersbusiness': '@ReutersBiz',
    'theeconomist': '@TheEconomist',
    'bbcbusiness': '@BBCBusiness',
    'cnbc': '@CNBC'
}

# function to return a range of dates
def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)

# function to scrape twitter posts in general from a specific account
def scrape_twitter(target, start_date, end_date, limit=10000):
    query_str = f'from:{tar_acs[target]} since:{str(start_date)} until:{str(end_date)}'
    twt_generator = twitter.TwitterSearchScraper(query_str).get_items()
    sliced_scraped_tweets = itertools.islice(twt_generator, limit)
    return pd.DataFrame(sliced_scraped_tweets)

# function to scrape twitter posts in batch from a specific account
def scrape_twitter_batch_and_save(target, start_date, end_date, wait_time=2, daily_limit=10000):
    dates = [single_date for single_date in daterange(start_date, end_date+timedelta(1))]
    for i in trange(len(dates)-1, desc=target):
        file_name = f'{target}_{str(dates[i])}.csv'
        file_path = os.path.join(data_dir, target, file_name)
        # only download tweet if not yet downloaded
        if not os.path.isfile(file_path):
            df = scrape_twitter(target, dates[i], dates[i+1], limit=daily_limit)
            df.to_csv(file_path, index=None)
            time.sleep(wait_time)
            
# ensure data director folders exist
data_dir = 'twitter/'
targets = list(tar_acs.keys())
for name in targets:
    tar_dir = os.path.join(data_dir, name)
    if not os.path.isdir(tar_dir):
        os.mkdir(tar_dir)            
       
for i in trange(len(targets), desc='Targets'):
    scrape_twitter_batch_and_save(targets[i], date(2010, 1, 1), date(2019, 1, 1))