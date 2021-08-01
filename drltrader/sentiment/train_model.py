import argparse
import time
import gc

import torch
import torch.nn as nn
from transformers import BertModel
import matplotlib.pyplot as plt

from .lib.data import SentimentData
from .lib.models import BERTLSTMSentimentModel, BERTGRUSentimentModel
from .lib.utils import Tracker, get_latest_model_path

def train_epoch(model, sentimentdata, loss_function, optimizer):
    total_loss=0
    correct_count=0
    model.train()
    for xi, yi in sentimentdata.train_dataloader:
        optimizer.zero_grad()
        xi = xi.to(device)
        yi = yi.to(device)
        y_hat = model(xi)
        loss = loss_function(y_hat, yi)
        total_loss += loss.detach()
        loss.backward()
        optimizer.step()
        correct_count += (torch.argmax(y_hat,axis=1)==yi).float().sum()
    return total_loss/sentimentdata.train_size, correct_count/sentimentdata.train_size
    
def evaluate(model, sentimentdata, loss_function):
    total_loss=0
    correct_count=0
    model.eval()
    for xi, yi in sentimentdata.test_dataloader:
        xi = xi.to(device)
        yi = yi.to(device)
        y_hat = model(xi)
        loss = loss_function(y_hat, yi)
        total_loss += loss
        correct_count += (torch.argmax(y_hat,axis=1)==yi).float().sum()
    return total_loss/sentimentdata.test_size, correct_count/sentimentdata.test_size

def plot_losses(train_loss, val_loss):
    _, ax = plt.subplots(figsize=(15,10))
    ax.plot(train_loss,
            color='r')
    ax.plot(val_loss,
            color='b')
    plt.legend(['train', 'val'])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss value over epoch')
    plt.show()
    plt.savefig('loss.png')

def plot_accs(train_acc, val_acc):
    _, ax = plt.subplots(figsize=(15,10))
    ax.plot(train_acc,
            color='r')
    ax.plot(val_acc,
            color='b')
    plt.legend(['train', 'val'])
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy over epoch')
    plt.show()
    plt.savefig('acc.png')

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="../data/FinancialPhraseBank-v1.0/all-data.csv",
                        help="training data file")
    parser.add_argument("--cuda", help="Enable cuda", default=True)
    parser.add_argument("--model", default="gru",
                        help="Must be either 'gru' or 'lstm'")
    parser.add_argument("--hidden_dims", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=1)
    parser.add_argument("--bidirectional", default=True)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--batch_size", default=4)
    parser.add_argument("--epoch", default=20)
    parser.add_argument("--retrain", default=False)
    parser.add_argument("--plot", default=True)
    
    args = parser.parse_args()
    
    if args.model not in ['lstm', 'gru']:
        raise NameError
    
    device = torch.device("cuda" if args.cuda else "cpu")
    
    sentimentdata = SentimentData(batch_size=args.batch_size)
    train_dataloader, test_dataloader = sentimentdata.process_train_test_data(args.data)
    print('Data is loaded.')
    
    bert = BertModel.from_pretrained('bert-base-uncased')
    print('BertModel-Base is downloaded and ready')
    
    if args.model == 'gru':
        model = BERTGRUSentimentModel(bert,
                                        args.hidden_dims,
                                        3,
                                        args.n_layers,
                                        args.bidirectional,
                                        args.dropout)
    elif args.model == 'lstm':
        model = BERTLSTMSentimentModel(bert,
                                        args.hidden_dim,
                                        3,
                                        args.n_layers,
                                        args.bidirectional,
                                        args.dropout)
    
    if args.retrain:
        latest_model_path = get_latest_model_path()
        model.load_state_dict(torch.load(latest_model_path))    
    
    # freezing the weights
    print(f'The model originally has {model.count_parameters():,} trainable parameters')

    for name, param in model.named_parameters():                
        if name.startswith('bert'):
            param.requires_grad = False

    print(f'After set to transfer learning, the model has {model.count_parameters():,} trainable parameters')
    print('\n**Training will be needed in the layers below:')
    for name, param in model.named_parameters():                
        if param.requires_grad:
            print(name)
            
    model = model.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.01)
    
    epoch_count = 0
    tracker = Tracker(load=args.retrain)
    
    EPOCHS = args.epoch
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    best_val_acc = float('-inf')
    
    for i in range(EPOCHS):
        total_loss = 0
        start = time.time()
        train_loss, train_acc = train_epoch(model, sentimentdata, loss_function, optimizer)
        val_loss, val_acc = evaluate(model, sentimentdata, loss_function)
        print(f'Epoch {epoch_count} | train loss: {train_loss:.4f} acc: {train_acc:.4f}' + 
            f' | val loss: {val_loss:.4f} acc: {val_acc:.4f}' + 
            f' | time taken: {time.time()-start:.2f}')

        tracker.record_info([float(metric) for metric in [epoch_count, train_loss, train_acc, val_loss, val_acc]])

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        epoch_count += 1

        if best_val_acc < val_acc: 
            torch.save(model.state_dict(), f'model/sentiment_model_best_epoch_{i}.pth')
            best_val_acc = val_acc
        elif epoch_count % 2 == 0:
            torch.save(model.state_dict(), f'model/sentiment_model_bk_epoch_{i}.pth')

        gc.collect()
        torch.cuda.empty_cache()
    
