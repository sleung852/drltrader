import csv
import glob

class Tracker:
    def __init__(self, file_name='tracker.csv', load=True):
        self.file_name = file_name
        if not load:
            tracker = open(file_name, 'w+')
            writer = csv.writer(tracker)
            writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])
            tracker.close()
    
    def record_info(self, info):
        with open(self.file_name, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(info)
            
def get_latest_model_path():
    paths = (glob.glob('model/sentiment_model_best_epoch_*.pth'))
    epochs = [int((p.split('.')[0]).split('_')[-1]) for p in paths]
    max_epoch = max(epochs)
    return f'model/sentiment_model_best_epoch_{max_epoch}.pth'
        