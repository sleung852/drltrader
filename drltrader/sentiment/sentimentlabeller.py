import argparse
import os
import sys

# insert the "import_test" directory into the sys.path
sys.path.insert(1, os.path.abspath(".."))

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import BertModel, BertTokenizer

# from lib.data import SentimentData
from .lib.models import BERTGRUSentimentModel
from .lib.utils import get_latest_model_path

class SentimentLabeller:
    def __init__(self):
        self._load_model()
        self._load_tokenizer()
        
        self.INV_LABEL_MAPPING = {
            0: 'negative',
            1: 'neutral',
            2: 'positive'
        }
        
    def _load_model(self):
        bert = BertModel.from_pretrained('bert-base-uncased')
        self.model = BERTGRUSentimentModel(bert, 256, 3, 1, True, 0.25)
        latest_model_path = get_latest_model_path()
        self.model.load_state_dict(torch.load(latest_model_path))
        self.model.eval()
        
    def _load_tokenizer(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_input_length = self.tokenizer.max_model_input_sizes['bert-base-uncased']
        
    def tokenize_and_cut(self, sentence):
        tokens = self.tokenizer.tokenize(sentence)
        tokens = tokens[:self.max_input_length-2]
        return tokens    
    
    def label_text(self, text, mode='num'):
        assert mode in ['label', 'num'], "mode must be either 'label' or 'num'"
        init_token_idx = self.tokenizer.cls_token_id
        eos_token_idx = self.tokenizer.sep_token_id
        pad_token_idx = self.tokenizer.pad_token_id
        
        X = [torch.tensor([init_token_idx] + [self.tokenizer.convert_tokens_to_ids(token) for token in self.tokenize_and_cut(text)] + [eos_token_idx], dtype=torch.long)]
        X = pad_sequence(X, padding_value=pad_token_idx).T
        
        with torch.no_grad():
            result = self.model(X)
            
        result = int(torch.argmax(result[0]))
        
        if mode == 'num':
            return result
        return self.INV_LABEL_MAPPING[result]
    
    def batch_label_text(self, text_list, mode='num'):
        assert mode in ['label', 'num'], "mode must be either 'label' or 'num'"
        init_token_idx = self.tokenizer.cls_token_id
        eos_token_idx = self.tokenizer.sep_token_id
        pad_token_idx = self.tokenizer.pad_token_id
        
        X = [torch.tensor([init_token_idx] + [self.tokenizer.convert_tokens_to_ids(token) for token in self.tokenize_and_cut(text)] + [eos_token_idx], dtype=torch.long) for text in text_list]
        X = pad_sequence(X, padding_value=pad_token_idx).T
        
        with torch.no_grad():
            result = self.model(X)
            
        if mode == 'num':
            return torch.argmax(result, dim=1).numpy()
        return [self.INV_LABEL_MAPPING[i] for i in torch.argmax(result, dim=1).numpy()]
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('text', metavar='text', type=str, help="Input text for sentiment score")
    parser.add_argument('--label', default=True)
    
    args = parser.parse_args()
    
    sl = SentimentLabeller()
    if args.label:
        print(sl.label_text(args.text, mode='label'))
    else:
        print(sl.label_text(args.text, mode='num'))
