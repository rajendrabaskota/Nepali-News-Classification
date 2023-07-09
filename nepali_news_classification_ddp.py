import os
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn import functional as F

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group


def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    
    init_process_group(backend='nccl', rank=rank, world_size=world_size)

def _tokenize_text(X_train, X_test, y_train, y_test):
    vectorizer = TfidfVectorizer(max_features=20000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    X_train_tfidf = torch.from_numpy(X_train_tfidf.toarray()).type(torch.FloatTensor)
    X_test_tfidf = torch.from_numpy(X_test_tfidf.toarray()).type(torch.FloatTensor)
    y_train = torch.from_numpy(y_train.to_numpy())
    y_test = torch.from_numpy(y_test.to_numpy())

    return X_train_tfidf, X_test_tfidf, y_train, y_test

def prepare_data():
    df = pd.read_csv("/kaggle/input/nepali-news-cleaned-2/nepali_news_cleaned_2.csv")
    all_labels = df['labels'].unique().tolist()
    n_labels = len(all_labels)

    id2label = {key: val for key, val in enumerate(all_labels)}
    label2id = {val: key for key, val in id2label.items()}
    df['labels'] = df['labels'].map(label2id)

    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['labels'])

    X_train = train_df['news']
    X_test = test_df['news']
    y_train = train_df['labels']
    y_test = test_df['labels']

    X_train_tfidf, X_test_tfidf, y_train, y_test = _tokenize_text(X_train, X_test, y_train, y_test)

    return X_train_tfidf, X_test_tfidf, y_train, y_test, n_labels


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
#         super().__init__()
        self.x = x
        self.y = y
    
    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def prepare_dataloader(dataset, batch_size):
    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            pin_memory=True,
                                            sampler=DistributedSampler(dataset))
    return dataloader


class NN_model(nn.Module):
    def __init__(self, input_units, output_units):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_units, 12000),
            nn.Linear(12000, 12000),
            nn.Linear(12000, 12000),
            nn.Linear(12000, output_units*4),
            nn.Linear(output_units*4, output_units)
        )
        
    def forward(self, x, y):
        logits = self.network(x)
        loss = F.cross_entropy(logits, y)
        
        return logits, loss


class Trainer():
    def __init__(self, rank, world_size, model, optimizer, epochs, batch_size):
        self.rank = rank
        self.world_size = world_size
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size

    def train(self, X_train_tfidf, X_test_tfidf, y_train, y_test):
        train_dataset = CustomDataset(X_train_tfidf, y_train)
        test_dataset = CustomDataset(X_test_tfidf, y_test)
        train_dl = prepare_dataloader(dataset=train_dataset, batch_size=self.batch_size)
        test_dl = prepare_dataloader(dataset=test_dataset, batch_size=self.batch_size)
        self.model = self.model.to(self.rank)
        self.model = DDP(self.model, device_ids=[self.rank])
        train_accuracies = torch.tensor([], device=self.rank)
        test_accuracies = torch.tensor([], device=self.rank)
        
        for epoch in range(self.epochs):
            train_accuracy = []
            print(f"Epoch: {epoch}")
            for data, target in tqdm(train_dl):
                logits, loss = self.model(data.to(self.rank), target.to(self.rank))
                preds = logits.argmax(-1)
                correct_predictions = torch.sum(target.to(self.rank) == preds).item() #torch.sum results a tensor object. .item() extracts the number
                train_accuracy.append(correct_predictions/len(preds))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
            x = sum(train_accuracy)/len(train_accuracy)
            train_accuracies = torch.cat((train_accuracies, torch.tensor([x], device=self.rank)))
            test_accuracy = self.compute_accuracy(test_dl)
            test_accuracies = torch.cat((test_accuracies, torch.tensor([test_accuracy], device=self.rank)))

        dist.reduce(train_accuracies, dst=0) #dist.reduce() gives the sum of value among different GPUs
        dist.reduce(test_accuracies, dst=0) # for eg. if GPU1: [1, 2, 3] GPU2: [2, 3, 4], then dist.reduce() returns [3, 5, 7]
        if dist.get_rank() == 0:
            print(f"train accuracies: {train_accuracies/self.world_size}") # gives the average accuracy among different GPUs
            print(f"test accuracies: {test_accuracies/self.world_size}")
            
        return train_accuracies/self.world_size, test_accuracies/self.world_size # returning the average accuracy
    
    @torch.no_grad()
    def compute_accuracy(self, test_dl):
        self.model.eval()
        test_accuracies = []
        for data, target in test_dl:
            logits, loss = self.model(data.to(self.rank), target.to(self.rank))
            preds = logits.argmax(-1)
            preds = preds.to(self.rank)
            x = torch.sum(target.to(self.rank) == preds).item()
            test_accuracies.append(x/len(preds))
            
        self.model.train()
        return sum(test_accuracies)/len(test_accuracies)

# def plot_graph(train_accuracies, test_accuracies):
#     epochs = torch.tensor(range(1, len(train_accuracies) + 1))

#     plt.plot(epochs.cpu(), train_accuracies.cpu(), 'b', label='Train Accuracy')
#     plt.plot(epochs.cpu(), test_accuracies.cpu(), 'r', label='Test Accuracy')
#     plt.title('Accuracy vs Epoch')
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy')
#     plt.legend()
#     plt.show()

def main(rank, world_size, epochs, batch_size, X_train_tfidf, X_test_tfidf, y_train, y_test, n_labels):
    ddp_setup(rank, world_size)
    # X_train_tfidf, X_test_tfidf, y_train, y_test, n_labels = prepare_data()
    n_embed = X_train_tfidf.shape[1]
    print(f"n_embed: {n_embed}")
    model = NN_model(input_units=n_embed, output_units=n_labels)
    print(f"Total number of parameters: {sum(p.numel() for p in model.parameters())/1e6} M")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    trainer = Trainer(rank, world_size, model, optimizer, epochs, batch_size)
    train_accuracies, test_accuracies = trainer.train(X_train_tfidf, X_test_tfidf, y_train, y_test)

    # if rank == 0:
    #     plot_graph(train_accuracies, test_accuracies)

    destroy_process_group()

if __name__ == "__main__":
    epochs = 10
    batch_size = 1024
    world_size = torch.cuda.device_count()
    X_train_tfidf, X_test_tfidf, y_train, y_test, n_labels = prepare_data()
    mp.spawn(main, args=(world_size, epochs, batch_size, X_train_tfidf, X_test_tfidf, y_train, y_test, n_labels), nprocs=world_size)
