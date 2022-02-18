from sklearn.metrics import confusion_matrix
from transformers import TFDistilBertForSequenceClassification, TFTrainer, TFTrainingArguments
import tensorflow as tf
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast
import pandas as pd
df2 = pd.read_csv('./smscollection.csv', sep='\t', names=["label", "message"])
X = list(df2['message'])
y = list(df2['label'])

# convert labels into 0's and 1's
y = list(pd.get_dummies(y, drop_first=True)['spam'])

# splitting the data into test and train
X_Train, X_Test, y_Train, y_Test = train_test_split(
    X, y, test_size=0.3, random_state=0)

# loading the model
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

# converts inputs into numerical arrays
train_encodings = tokenizer(X_Train, truncation=True, padding=True)
test_encodings = tokenizer(X_Test, truncation=True, padding=True)


train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    y_Train
))

test_dataset = tf.data.Dataset.from_tensor_slices((
    dict(test_encodings),
    y_Test
))


training_args = TFTrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=4,   # batch size per device during training
    per_device_eval_batch_size=8,    # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)

with training_args.strategy.scope():
    model = TFDistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased")

trainer = TFTrainer(
    # the instantiated 🤗 Transformers model to be trained
    model=model,
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=test_dataset            # evaluation dataset
)

trainer.train()
trainer.evaluate(test_dataset)
trainer.predict(test_dataset)
output = trainer.predict(test_dataset)[1]

confusionMatrixResult = confusion_matrix(y_Test, output)

trainer.save_model('Transformer_Bert_Model')

improved_model = TFDistilBertModel.from_pretrained("Transformer_Bert_Model")

