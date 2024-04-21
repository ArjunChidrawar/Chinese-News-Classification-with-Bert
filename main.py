import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from data_preprocessing import *



bert_model = BertModel.from_pretrained(chinese_bert, return_dict = False)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
# last_hidden_state, pooled_output = bert_model(
#     input_ids = encoding['input_ids'],
#     attention_mask = encoding['attention_mask']
# )


##Building Sentiment Classifier
class NewsClassifier(nn.Module):
    def __init__(self, n_classes):
        super(NewsClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(chinese_bert, return_dict = False)
        self.dropout = nn.Dropout(p = 0.3) #dropout layers
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes) #linear transformation on input
        self.softmax = nn.Softmax(dim = 1) #apply activation function (converts scores into probabilities)
        #^ helps us gauge model's confidence in a given classification
    
    def forward(self, input_ids, attention_mask):
        last_hidden_state, pooled_output = self.bert(
            input_ids = input_ids,
            attention_mask = attention_mask
        )
        output = self.dropout(pooled_output)
        output = self.out(output)
        return self.softmax(output)

model = NewsClassifier(5).to(device)
input_ids = data['input_ids'].to(device)
attention_mask = data['attention_mask'].to(device)

# print(input_ids.shape) #prints batch_size, number in batch
# print(attention_mask.shape)
# print(model(input_ids, attention_mask)) #prints a 2-d array of classification probabilities

##Training the Model
optimizer = torch.optim.AdamW(model.parameters(), lr = 2e-5)
total_steps = len(train_dataloader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps = 0,
    num_training_steps= total_steps
)
loss_fn = nn.CrossEntropyLoss().to(device)
def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0
    batch = 1
    batches = len(data_loader)
    for d in data_loader:
        input_ids = d['input_ids'].to(device)
        attention_mask = d['attention_mask'].to(device)
        targets = d['targets'].to(device)

        outputs = model(
            input_ids = input_ids,
            attention_mask = attention_mask
        )

        _, preds = torch.max(outputs, dim = 1)
        loss = loss_fn(outputs, targets)
        
        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        print(f'Batch # {batch}/{batches}')
        batch += 1
    return (correct_predictions.double()/n_examples), np.mean(losses)

#test model -- same as train but we are now evaluating performance, so no gradient and no optimization
def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids = d['input_ids'].to(device)
            attention_mask = d['attention_mask'].to(device)
            targets = d['targets'].to(device)

            outputs = model(
                input_ids = input_ids,
                attention_mask = attention_mask
            )

            _, preds = torch.max(outputs, dim = 1)
            loss = loss_fn(outputs, targets)
        
            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())

    return (correct_predictions.double()/n_examples), np.mean(losses)



#training over specified number of EPOCHS

def main():
    history = defaultdict(list)
    best_accuracy = 0

    for e in range(EPOCHS):

        print(f'Epoch {e+1}/{EPOCHS}')
        print('-'*10)

        train_acc, train_loss = train_epoch(model, train_dataloader, loss_fn, optimizer, device, scheduler, len(train_dataset)) 

        print(f'Train Loss: {train_loss}, Accuracy: {train_acc}')

        val_acc, val_loss = eval_model(model, val_dataloader, loss_fn, device, len(val_dataset))

        print(f'Validation Loss: {val_loss}, Accuracy: {val_acc}')

        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
    
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)

        if val_acc > best_accuracy:
            torch.save(model.state_dict(), 'best_model_state.bin')
            best_accuracy = val_acc

#COMMENT THIS OUT AFTER TRAINING THE MODEL
# if __name__ == "__main__":
#     main()
    