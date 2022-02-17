from sklearn.metrics import confusion_matrix
from transformers import TFDistilBertForSequenceClassification, TFTrainer, TFTrainingArguments
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast
from textGen import generate

import tensorflow as tf
import pandas as pd


df2 = pd.read_csv('./smscollection.csv', sep='\t', names=["label", "message"])
X = list(df2['message'])
y = list(df2['label'])

# convert labels into 0's and 1's
y = list(pd.get_dummies(y, drop_first=True)['spam'])

for index, row in df2.iterrows():
    print(generate(row['message']))
    print("******************")
