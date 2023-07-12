# Basic BERT Topic Classification

## Training

```py
python bert_train_topic_model.py
```

This Python script trains a BERT-based topic classification model using the 20 Newsgroups dataset. The script first loads the dataset using Scikit-learn's fetch_20newsgroups function and preprocesses it into a Pandas DataFrame. The script then splits the dataset into training and validation sets, and tokenizes the text using the BERT tokenizer.

The script defines a custom PyTorch Dataset class to load the tokenized data into memory, and instantiates a BERT-based sequence classification model using the BertForSequenceClassification class from the Transformers library. The script then trains the model using the Trainer class from the Transformers library, and saves the resulting model to disk.

The script also saves the class names to a pickle file for later use in predicting the topic of new text documents. The script can be configured by modifying the TrainingArguments object, which controls the training hyperparameters such as the number of epochs, batch size, and learning rate.

## Inference

```py
python bert_topic_model_inference.py
```

This file, bert_topic_model_inference.py, contains code for performing inference using a pre-trained BERT model for text classification. The code loads a pre-trained BERT model and class names from files, and defines a predict function that takes in a text input, tokenizes it using the BERT tokenizer, and returns the predicted class name for the input text. The code also includes an example usage of the predict function, where it predicts the class name for a sample text input.
