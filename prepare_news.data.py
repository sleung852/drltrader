import argparse
import logging

from tqdm import tqdm

import pandas as pd

if __name__ == '__main__':
    
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', type=str, default='aapl')
    parser.add_argument('--files_path_format', type=str, default='data/{}_finance_data_{}.csv') 
    parser.add_argument('--news_file_path', type=str, default='data/all_combined_news_cleaned_labelled.csv')
    args = parser.parse_args()
    
    train = pd.read_csv(args.files_path_format.format(args.ticker, 'train'))
    eval = pd.read_csv(args.files_path_format.format(args.ticker, 'val'))
    test = pd.read_csv(args.files_path_format.format(args.ticker, 'test'))
    
    train_len = train.shape[0]
    eval_len = eval.shape[0] + train_len
    time_v = list(train.time.values) + list(eval.time.values) + list(test.time.values)
    del train, eval, test
    
    news = pd.read_csv(args.news_file_path)
    news.sort_values('date', inplace=True)
    news = news[(news.date>=time_v[0]) & (news.date <= time_v[-1])]
    news.reset_index(drop=True, inplace=True)
    
    idx = 1
    pos = [0]
    neg = [0]
    neu = [0]

    neg_l = 0 #0
    neu_l = 0 #1
    pos_l = 0 #2

    ind_n = 0
    idx = 1

    pbar = tqdm(total = news.index[-1])

    # Runtime is O(N) ~ fastest possible
    while ind_n < news.index[-1]-1 or idx < len(time_v)-1:
        try:
            date = news.date.iloc[ind_n]
            if date > time_v[idx]:
                neg.append(neg_l)
                neu.append(neu_l)
                pos.append(pos_l)
                neg_l = 0 #0
                neu_l = 0 #1
                pos_l = 0 #2
                idx += 1
                
            else:
                if news.sentiment.iloc[ind_n] == 0:
                    neg_l += 1
                elif news.sentiment.iloc[ind_n] == 1:
                    neu_l += 1
                elif news.sentiment.iloc[ind_n] == 2:
                    pos_l += 1
                else:
                    raise ValueError
                ind_n += 1
                pbar.update(1)
        except:
            print(idx, ind_n)
            break
    pbar.close()
    
    news = pd.DataFrame(
        {
            'time': time_v,
            'pos': pos + [0]*(len(time_v)-len(pos)),
            'neu': neu + [0]*(len(time_v)-len(pos)),
            'neg': neg + [0]*(len(time_v)-len(pos)),
        }
    )
    
    sum_col = news[['pos', 'neu', 'neg']].sum(axis=1)
    for col in ['pos', 'neu', 'neg']:
        news[col] = news[col]/sum_col
    news.fillna(value=0, inplace=True)
    news['freq'] = sum_col/300
    
    news[:train_len].to_csv('data/news_train.csv',index=False)
    news[train_len:eval_len].to_csv('data/news_eval.csv',index=False)
    news[eval_len:].to_csv('data/news_test.csv',index=False)