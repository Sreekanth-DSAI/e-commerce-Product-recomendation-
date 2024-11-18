"""
Fine-tuning a GPT-2 Model for Product Recommendation Sentiment Classification

This script fine-tunes a pre-trained GPT-2 model to classify customer reviews as "positive" 
or "negative" based on sentiment, using review text data. After fine-tuning, the model is 
saved locally for deployment in a recommendation system. 

Steps:
1. Load and preprocess data from a JSON file.
2. Tokenize the data and convert labels for sentiment classification.
3. Define and configure a custom `Trainer` for model fine-tuning.
4. Train and evaluate the model, calculating BLEU score for evaluation.
5. Save the fine-tuned model and tokenizer for later deployment.
"""

# Import necessary libraries
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset
import torch
import json

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load Data from JSON File
with open("dataset.json", "r") as f:
    data = [json.loads(line) for line in f]

# Convert JSON data to a DataFrame for easier manipulation
df = pd.DataFrame(data)

# Select relevant columns for fine-tuning
df["recommendation"] = df["overall"].apply(lambda x: "positive" if x >= 4 else "negative")

# Drop rows with empty 'reviewText' entries to ensure clean input
df = df.dropna(subset=["reviewText"])

df = df.iloc[:1000,] #Sample data for training
 
# Convert to Hugging Face Dataset and split into train/test sets
dataset = Dataset.from_pandas(df[["reviewText", "recommendation"]])
dataset = dataset.train_test_split(test_size=0.2)

# Load Pre-trained Model and Tokenizer
model_name = "gpt2"  # preferred model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# Add padding token if tokenizer doesn't have one
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

# Custom Trainer Class to compute custom loss function
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Override Trainer's compute_loss function to apply custom loss calculation.
        
        Args:
            model (torch.nn.Module): The fine-tuning model.
            inputs (dict): Input data for training, including 'labels'.
            return_outputs (bool): Whether to return outputs with loss.

        Returns:
            torch.Tensor: The calculated loss, and optionally the outputs.
        """
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Define loss function for classification
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

# Preprocess Function for Tokenization and Labeling
def preprocess_function(examples):
    """
    Tokenize review text and recommendation labels, truncating and padding to fixed length.

    Args:
        examples (dict): Dictionary with 'reviewText' and 'recommendation'.

    Returns:
        dict: Tokenized inputs and labels with input IDs.
    """
    inputs = tokenizer(examples["reviewText"], padding="max_length", truncation=True, max_length=128)
    inputs["labels"] = tokenizer(examples["recommendation"], padding="max_length", truncation=True, max_length=128)["input_ids"]
    return inputs

# Tokenize Dataset for Training and Evaluation
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Define Training Arguments for Fine-Tuning
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=10,
    weight_decay=0.01,
    logging_dir="./logs",
    load_best_model_at_end=True,
)

# Initialize Custom Trainer for Fine-Tuning
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)

# Start Model Training
trainer.train()

# Optional Evaluation: Generate Recommendations and Calculate BLEU Score
from nltk.translate.bleu_score import sentence_bleu

def generate_recommendation(text):
    """
    Generates a recommendation based on the input text using the fine-tuned model.
    
    Args:
        text (str): Review text input.

    Returns:
        str: Generated recommendation.
    """
    inputs = tokenizer.encode(text, return_tensors="pt").to(device)
    outputs = model.generate(inputs, max_length=1000, num_return_sequences=1)
    recommendation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return recommendation

# Calculate BLEU Score for Model Performance
references = [example["recommendation"] for example in tokenized_datasets["test"]]
predictions = [generate_recommendation(text) for text in tokenized_datasets["test"]["reviewText"]]

# Average BLEU Score Calculation
bleu_scores = [sentence_bleu([ref], pred) for ref, pred in zip(references, predictions)]
average_bleu = sum(bleu_scores) / len(bleu_scores)
print(f"Average BLEU Score: {average_bleu}")

# Save Fine-Tuned Model and Tokenizer for Deployment
model.save_pretrained("fine_tuned_model")
tokenizer.save_pretrained("fine_tuned_model")
