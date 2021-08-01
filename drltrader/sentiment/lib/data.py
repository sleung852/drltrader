import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, WeightedRandomSampler

from transformers import BertTokenizer

class SentimentData:
    def __init__(self, batch_size=4):
        
        
        # map label
        self.LABEL_MAPPING = {
            'negative': 0,
            'neutral': 1,
            'positive': 2
        }
        self.INV_LABEL_MAPPING = {val: label for label, val in self.LABEL_MAPPING.items()}
        
        self.BATCH_SIZE = batch_size
        self.load_tokenizer()

    def load_tokenizer(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_input_length = self.tokenizer.max_model_input_sizes['bert-base-uncased']

    def load_and_clean_csv(self, csv_file):
        df = pd.read_csv(csv_file, header= None)
        df.columns=['label', 'text']
        df['label'] = df['label'].astype('category')
        return df

    #from sklearn.utils import class_weight # alternative
    def create_sample_weights(self, y_train):
        y_train_size = len(y_train)
        freq_record = {key: 0 for key in self.LABEL_MAPPING.values()}
        # count freq of each class
        for i in y_train:
            freq_record[int(i.numpy())] += 1
        # output placeholder
        y_train_weights = np.zeros(len(y_train))
        for i in range(y_train_weights.shape[0]):
            y_train_weights[i] = 1. / freq_record[int(y_train[i])]
    #         y_train_weights[i]= 0.6667 - freq_record[int(y_train[i])]/y_train_size
        return y_train_weights
    
    def tokenize_and_cut(self, sentence):
        tokens = self.tokenizer.tokenize(sentence)
        max_input_length = self.tokenizer.max_model_input_sizes['bert-base-uncased']
        tokens = tokens[:self.max_input_length-2]
        return tokens

    def process_train_test_data(self, csv_file):
        df = self.load_and_clean_csv(csv_file)
        
        init_token_idx = self.tokenizer.cls_token_id
        eos_token_idx = self.tokenizer.sep_token_id
        pad_token_idx = self.tokenizer.pad_token_id
        
        y = [torch.tensor(self.LABEL_MAPPING[label], dtype=torch.long) for label in df['label'].values]
        X = [torch.tensor([init_token_idx] + [self.tokenizer.convert_tokens_to_ids(token) for token in self.tokenize_and_cut(text)] + [eos_token_idx], dtype=torch.long) for text in df['text'].values]
        X = pad_sequence(X, padding_value=pad_token_idx).T
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        test_data = [(xi, yi) for xi, yi in zip(X_test, y_test)]
        train_data = [(xi, yi) for xi, yi in zip(X_train, y_train)]
        
        self.train_size = len(train_data)
        self.test_size = len(test_data)
                
        sample_weights = self.create_sample_weights(y_train)
        # add sampler to tackle imbalance data issue
        weight_sampler = WeightedRandomSampler(sample_weights, len(y_train), replacement=True)
        
        self.train_dataloader = DataLoader(train_data, batch_size=self.BATCH_SIZE, sampler=weight_sampler)
        self.test_dataloader = DataLoader(test_data, batch_size=self.BATCH_SIZE, shuffle=True)
        return self.train_dataloader, self.test_dataloader
    
    def process_data(self, csv_file, column_name):
        df = pd.read_csv(csv_file)
        text_col = df[column_name].values
        
        init_token_idx = self.tokenizer.cls_token_id
        eos_token_idx = self.tokenizer.sep_token_id
        pad_token_idx = self.tokenizer.pad_token_id
        
        X = [torch.tensor([init_token_idx] + [self.tokenizer.convert_tokens_to_ids(token) for token in self.tokenize_and_cut(text)] + [eos_token_idx], dtype=torch.long) for text in df['text'].values]
        X = pad_sequence(X, padding_value=pad_token_idx).T
        
        return X