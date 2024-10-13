from train import train_model
from evaluate import evaluate_model

dataset_path = './data/train.csv'


test_df = train_model(dataset_path)
print("Training completed.")

accuracy, report = evaluate_model(test_df)
print(f"Evaluation completed. Accuracy: {accuracy:.2f}")