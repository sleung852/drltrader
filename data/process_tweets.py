import pandas as pd
import os
import glob
from tqdm import tqdm
import shutil

news_data_mother_dir = 'twitter'
news_data_child_dirs = os.listdir(news_data_mother_dir)

old_csv_files = glob.glob(os.path.join(news_data_mother_dir, '*.csv'))

for old_csv_file in old_csv_files:
    new_path = shutil.move(old_csv_file, 'twitter_old')
    
news_data_child_dirs = os.listdir(news_data_mother_dir)

tar_news_data_dir = os.listdir(os.path.join(news_data_mother_dir, news_data_child_dirs[0]))

for tw_acc in tqdm(news_data_child_dirs, total=len(news_data_child_dirs)):
    print(f'start working with {tw_acc}')
    file_list = os.listdir(os.path.join(news_data_mother_dir, tw_acc))
    file_len = len(file_list)
    ind = 0
    for i in range(len(file_list)):
        try:
            df = pd.read_csv(os.path.join(news_data_mother_dir, tw_acc, file_list[i]))
            ind = i
            break
        except:
            pass
    
    for i in range(ind, len(file_list)):
        try:
            df_temp = pd.read_csv(os.path.join(news_data_mother_dir, tw_acc, file_list[i]))
            df = pd.concat([df_temp, df])
        except:
            pass
        
    combined_file_name = f"{tw_acc}_combined.csv"
    print(f'saving file {combined_file_name} -> {news_data_mother_dir}')
    df.to_csv(os.path.join(news_data_mother_dir, combined_file_name), index=False, encoding='utf-8-sig')
    print(f'finished ')
    
# create a version that combined all data
file_list = glob.glob('data/twitter/'+'*.csv')

df = pd.read_csv(file_list[0])

for file in file_list[1:]:
    df_temp = pd.read_csv(file)
    df = pd.concat([df_temp, df])
    
df.to_csv(os.path.join(news_data_mother_dir, 'all_combined_news.csv'), index=False, encoding='utf-8-sig')