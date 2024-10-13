from transformers import AutoModelForSequenceClassification, BertTokenizer, Trainer
import torch
import json
import argparse

def load_tokenizer(tokenizer_path='./LLM_model/special_tokenizer'):
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path, do_lower_case=True, use_fast=True)
    return tokenizer

def load_trained_model(model_path, num_labels=49):
    base_model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)
    return base_model

def load_label_mapping(label_mapping_path):
    with open(label_mapping_path, "r") as f:
        label_mapping = json.load(f)
    label_mapping = {int(k): v for k, v in label_mapping.items()}
    return label_mapping

def predict(text, model, tokenizer, label_mapping):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding='max_length', max_length=512)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    if 'token_type_ids' in inputs and 'token_type_ids' not in model.config.to_dict():
        inputs.pop('token_type_ids')

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = logits.argmax().item()
    predicted_label = label_mapping[predicted_class_id]
    return predicted_label

def evaluate_model_on_test_set(tokenized_ds,model, tokenizer, model_name="bert_model"):
    print("\nTest Evaluation starts")

    trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="./results",
        per_device_eval_batch_size=4,
        eval_strategy="epoch",
        save_strategy="epoch",
    ),
    train_dataset=None,  # No training for evaluation
    eval_dataset=tokenized_ds['test'] ,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
    )
    
    results = trainer.evaluate()
    results_file = f"../results/{model_name}_test_evaluation.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"Test evaluation results saved to {results_file}")


if __name__ == "__main__":

    #take sentence from command line
    parser = argparse.ArgumentParser(description="BERT inference script for Turkish text classification.")
    parser.add_argument("text", type=str, help="The text to classify.")
    args = parser.parse_args()

    # Load tokenizer and model
    inference_tokenizer = load_tokenizer()
    label_mapping = load_label_mapping("./data/label_mapping.json")
    model_path = "./models/trained_BERT_model" 
    inference_model = load_trained_model(model_path, num_labels=len(label_mapping))

    predicted_label = predict(args.text, inference_model, inference_tokenizer, label_mapping)
    print(f"Predicted label: {predicted_label}")