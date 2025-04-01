from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load the saved model and tokenizer
model_path = './saved_model_25'
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# Ensure the model is in evaluation mode
model.eval()

# Token limit for BERT base model
MAX_TOKENS = 512

# Function to make predictions
def predict(text1, text2):
    # Tokenize the input text
    chunk1_tokens = tokenizer.tokenize(text1)
    chunk2_tokens = tokenizer.tokenize(text2)
    
    # Truncate chunk1 and chunk2 to fit within MAX_TOKENS
    chunk1_tokens = chunk1_tokens[-254:]  # Take last 254 tokens from chunk1
    chunk2_tokens = chunk2_tokens[:254]   # Take first 254 tokens from chunk2
    
    # Combine the tokens and add special tokens
    tokens = ['[CLS]'] + chunk1_tokens + ['[SEP]'] + chunk2_tokens + ['[SEP]']
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    
    # Create attention mask
    attention_mask = [1] * len(token_ids)
    
    # Pad token_ids and attention_mask to MAX_TOKENS
    padding_length = MAX_TOKENS - len(token_ids)
    token_ids += [tokenizer.pad_token_id] * padding_length
    attention_mask += [0] * padding_length
    
    # Convert to tensors
    input_ids = torch.tensor([token_ids])
    attention_mask = torch.tensor([attention_mask])
    
    # Make prediction
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    
    return predicted_class

# Example usage
text1 = "This is the first chunk of text."
text2 = "This is the second chunk of text."
prediction = predict(text1, text2)
print(f"Predicted class: {prediction}")


import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
data=pd.read_csv('output/complete_data.csv')
data['label'] = data['label'].astype(int)
y_true = []
y_pred = []
for i,row in data.sample(1000).iterrows():
    y_pred.append(predict(row['chunk1'], row['chunk2']))
    y_true.append(row['label'])


# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy}")

report = classification_report(y_true, y_pred, target_names=['Class 0', 'Class 1'])
print("Classification Report:")
print(report)
