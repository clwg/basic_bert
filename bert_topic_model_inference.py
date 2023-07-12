import torch
from transformers import BertTokenizerFast, BertForSequenceClassification
import pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

model = BertForSequenceClassification.from_pretrained('./bert_topic_model').to(device)

# Load class names from file
with open('class_names.pkl', 'rb') as f:
    class_names = pickle.load(f)

def predict(text):
    inputs = tokenizer(text, truncation=True, padding=True, max_length=512, return_tensors='pt').to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    predicted_class_idx = outputs.logits.argmax(-1).item()

    return class_names[predicted_class_idx]

text = "This is test text."
predicted_class_name = predict(text)

print(f'The predicted class of the text "{text}" is {predicted_class_name}.')
