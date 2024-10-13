from transformers import BertTokenizer, TrainingArguments, Trainer, AutoModelForSequenceClassification, DataCollatorWithPadding 
from evaluate_bert import compute_metrics
import torch
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
import json


class CustomTrainer(Trainer):
    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        labels = self.train_dataset['label']
        labels = np.array(labels)

        classes, class_counts = np.unique(labels, return_counts=True)
        class_weights = {cls: 1.0 / count for cls, count in zip(classes, class_counts)}

        sample_weights = np.array([class_weights[label] for label in labels])
        sample_weights = torch.from_numpy(sample_weights).float()

        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

        # Return DataLoader with the custom sampler
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
        )

def load_model():
    model = AutoModelForSequenceClassification.from_pretrained('dbmdz/distilbert-base-turkish-cased', num_labels=49)

    # Check if a GPU is available
    if not torch.cuda.is_available():
        # Freeze all model parameters (leaves classifier head trainable)
        for param in model.base_model.parameters():
            param.requires_grad = False
        print("\nRunning on CPU. Freezing base model parameters.")
    else:
        print("\nRunning on GPU. Keeping all parameters trainable.")
    
    trainable_params = [name for name, param in model.named_parameters() if param.requires_grad]
    print("\nTrainable parameters:", trainable_params)
    return model 

def create_trainer(model, tokenized_ds, tokenizer):
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=1,
        eval_strategy="epoch",
        logging_strategy="epoch",
        logging_dir='./logs',
        learning_rate=1e-4,
        save_strategy="epoch",   
    )
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["val"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    return trainer

def train_model(trainer):
    print("\nTrain starts")
    trainer.train()

def evaluate_model(trainer, model_name="bert_model"):
    print("\nEvaluation starts")
    results = trainer.evaluate()

    results_file = f"./results/{model_name}_evaluation.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"Evaluation results saved to {results_file}")

def save_model(trainer):
    trainer.save_model("./models/bertturk_model")

def load_saved_model():
    model = AutoModelForSequenceClassification.from_pretrained("./models/bertturk_model")
    return model