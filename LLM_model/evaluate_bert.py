import numpy as np 
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def print_conf(predictions, labels):

  # Assuming labels and predictions are your true and predicted values
  conf_matrix = confusion_matrix(labels, predictions)
  unique_labels = np.unique(labels)
  conf_matrix_df = pd.DataFrame(conf_matrix, index=unique_labels, columns=unique_labels)
  # Calculate the ratio of each label in the dataset
  counts = np.bincount(labels)  # Gives count for all classes from 0 to max(labels)
  label_ratios = [counts[label] / len(labels) for label in unique_labels]
  # Create a DataFrame with labels and their ratios
  label_info = pd.DataFrame({
      'Label': unique_labels,
      'Ratio': label_ratios
  })
  # Set display options for pandas to see the full matrix
  pd.set_option('display.max_rows', None)
  pd.set_option('display.max_columns', None)

  # Print the confusion matrix with labels and ratios
  print("Confusion Matrix with Class Labels and Ratios:")
  print(conf_matrix_df)

def compute_metrics(eval_pred):
    try:
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        average_type = 'weighted' #macro: each class independently, weighted: giving more weight to classes with more samples

        accuracy = (predictions == labels).mean()
        precision = precision_score(labels, predictions, average=average_type, zero_division=0)
        recall = recall_score(labels, predictions, average=average_type, zero_division=0)
        f1 = f1_score(labels, predictions, average=average_type,zero_division=0)

        average_type = 'macro' #macro: each class independently, weighted: giving more weight to classes with more samples

        precision2 = precision_score(labels, predictions, average=average_type, zero_division=0)
        recall2 = recall_score(labels, predictions, average=average_type, zero_division=0)
        f12 = f1_score(labels, predictions, average=average_type,zero_division=0)

        print(classification_report(labels, predictions))

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "precision_macro": precision2,
            "recall_macro": recall2,
            "f1_macro": f12
        }
    except Exception as e:
        print(f"Error computing metrics: {e}")
        return {}