from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict, Dataset
from sklearn.model_selection import train_test_split
import pandas as pd

# Load pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Token limit for BERT base model
MAX_TOKENS = 512

# Load dataset
data = pd.read_csv('output/complete_data.csv')

print(data.info())

# Ensure labels are integers
data['label'] = data['label'].astype(int)

# Split dataset into train and test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=8)

# Convert to Hugging Face dataset format
train_dataset = DatasetDict({"train": Dataset.from_pandas(train_data)})
test_dataset = DatasetDict({"test": Dataset.from_pandas(test_data)})

# Tokenize the dataset
def preprocess_function(examples):
    input_ids = []
    attention_masks = []
    labels = []

    for i in range(len(examples['chunk1'])):
        chunk1_text = examples['chunk1'][i]
        chunk2_text = examples['chunk2'][i]
        
        # Tokenize chunk1 and chunk2 separately
        chunk1_tokens = tokenizer.tokenize(chunk1_text)
        chunk2_tokens = tokenizer.tokenize(chunk2_text)
        
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
        
        input_ids.append(token_ids)
        attention_masks.append(attention_mask)
        labels.append(examples['label'][i])

    return {
        'input_ids': input_ids,
        'attention_mask': attention_masks,
        'labels': labels  # Ensure the key is 'labels' for Trainer compatibility
    }

tokenized_datasets = train_dataset.map(preprocess_function, batched=True)
tokenized_test_datasets = test_dataset.map(preprocess_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=25,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    save_steps=10,
    save_total_limit=2,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_test_datasets['test'],
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Evaluate the model
results = trainer.evaluate()
print(results)



# Save the trained model and tokenizer
model.save_pretrained('./saved_model_25')
tokenizer.save_pretrained('./saved_model_25')

print("Model and tokenizer saved to './saved_model'")
