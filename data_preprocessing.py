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
import lzma


df = pd.read_csv('reviews.csv')
df['Label'] = df['Label'] - 1 #Making sure that values range from 0 to 4 instead of 1 to 5
df = df[0:10000]
print(len(df))



#SHOW DISTRIBUTION BETWEEN CLASSES TO CHECK FOR IMBALANCE
# ax = sns.countplot(df.Label)
# plt.xlabel('review type')
# plt.show()

chinese_bert = 'bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(chinese_bert)

#Just for display purposes of how the tokenizer works


# tokens = tokenizer.tokenize(sample_txt)
# token_ids = tokenizer.convert_tokens_to_ids(tokens)
# print(f' Sentence: {sample_txt}')
# print(f'   Tokens: {tokens}')
# print(f'Token IDs: {token_ids}')
# encoding = tokenizer.encode_plus(
#     sample_txt,
#     max_length = 32,
#     add_special_tokens = True,
#     return_token_type_ids = False,
#     padding = 'max_length',
#     return_attention_mask = True,
#     return_tensors = 'pt'
# )

#Special Tokens: [PAD], [UNK], [CLS], [SEP]

#Choosing Sequence Length
# token_lens = []
# for txt in df.Content:
#     token = tokenizer.encode(txt, max_length = 512)
#     token_lens.append(len(token))
# print(max(token_lens))
# sns.displot(token_lens)
# plt.show()

#Vast majority of tokens are less than len of 250, so I set max_length to 250

#Create a Dataset
class newsDataset(Dataset):
    def __init__(self, content, target, tokenizer, max_len):
        self.content = content
        self.target = target
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.content)
    
    def __getitem__(self, item):
        content = str(self.content[item])
        
        encoding = tokenizer.encode_plus(
        content,
        max_length = self.max_len,
        add_special_tokens = True,
        truncation = True,
        return_token_type_ids = False,
        padding = 'max_length',
        return_attention_mask = True,
        return_tensors = 'pt'
        )
        return {
            'content': content,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(self.target[item], dtype = torch.long)
        }

MAX_LEN = 100
BATCH_SIZE = 128
EPOCHS = 10

#Creating train, test, validation sets
train_dataset, test_dataset = train_test_split(df, test_size= 0.2, random_state = 4)
test_dataset, val_dataset = train_test_split(test_dataset, test_size= 0.5, random_state = 4)

print(train_dataset.shape)
print(val_dataset.shape)
print(test_dataset.shape)


def create_data_loader(df, tokenizer, max_len, batch_size):
    dataset = newsDataset(
        content = df.Content.to_numpy(),
        target = df.Label.to_numpy(),
        tokenizer = tokenizer,
        max_len = max_len
    )
    return DataLoader(dataset, batch_size = batch_size, num_workers = 0)

train_dataloader = create_data_loader(train_dataset, tokenizer, MAX_LEN, BATCH_SIZE)
val_dataloader = create_data_loader(val_dataset, tokenizer, MAX_LEN, BATCH_SIZE)
test_dataloader = create_data_loader(test_dataset, tokenizer, MAX_LEN, BATCH_SIZE)

data = next(iter(train_dataloader))





#Making sure all the tensors are the same size
# for batch in train_dataloader:
#     input_ids = batch['input_ids']
#     attention_mask = batch['attention_mask']
#     targets = batch['targets']
   
#     if input_ids.size(1) > 250 or attention_mask.size(1) > 250 or targets.size(0) > 250:
#         print("Input IDs size:", input_ids.size())
#         print("Attention mask size:", attention_mask.size())
#         print("Targets size:", targets.size())

