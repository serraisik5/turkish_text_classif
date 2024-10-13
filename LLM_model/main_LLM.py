from transformers import BertTokenizer, TrainingArguments, Trainer
from data_preprocess import load_datasets, tokenize_datasets
from train_bert import create_trainer, train_model, load_model, save_model,load_saved_model, evaluate_model
from inference import load_tokenizer,load_trained_model, load_label_mapping, predict, evaluate_model_on_test_set

# Load datasets 
train_dataset, test_dataset, val_dataset, label_mapping = load_datasets()

# pre-trained tokenizer with added tokens
tokenizer = BertTokenizer.from_pretrained('./LLM_model/special_tokenizer')

# Tokenize datasets
tokenized_ds = tokenize_datasets(train_dataset, test_dataset, val_dataset, tokenizer)

model = load_model()

trainer = create_trainer(model, tokenized_ds, tokenizer)
train_model(trainer)

evaluate_model(trainer)
save_model(trainer)

# Evaluate On Test set
evaluate_model_on_test_set(tokenized_ds, model, tokenizer)

"""
# Example Inference

model_path = "../models/bertturk_model"
inference_tokenizer = load_tokenizer()

label_mapping = load_label_mapping("../data/label_mapping.json")
inference_model = load_trained_model(model_path, num_labels=len(label_mapping))

# Example inference
text = "Türkçe sondan eklemeli bir dildir."  
predicted_label = predict(text, inference_model, inference_tokenizer, label_mapping)
print(f"Predicted label: {predicted_label}")

"""