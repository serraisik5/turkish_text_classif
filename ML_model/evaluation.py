from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import os
import csv

def evaluate_metrics(y_true, y_pred, dataset_type="Train", results_file="./results/ML_evaluation_train_results.csv"):
    """
    Evaluates and prints the metrics for the given predictions and writes them to a CSV file.
    """
    print(f"Evaluation results for {dataset_type} dataset:")

    # macro metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average='macro')
    recall_macro = recall_score(y_true, y_pred, average='macro')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    
    # weighted metrics
    precision_weighted = precision_score(y_true, y_pred, average='weighted')
    recall_weighted = recall_score(y_true, y_pred, average='weighted')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    
    print(f"Accuracy: {accuracy}")
    print(f"Macro Precision: {precision_macro}")
    print(f"Macro Recall: {recall_macro}")
    print(f"Macro F1-Score: {f1_macro}")
    print(f"Weighted Precision: {precision_weighted}")
    print(f"Weighted Recall: {recall_weighted}")
    print(f"Weighted F1-Score: {f1_weighted}")
    print("------------------------------")
    

    class_report = classification_report(y_true, y_pred, output_dict=True)

    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    with open(results_file, mode='w', newline='') as file:  # Overwrite the file
        writer = csv.writer(file)
        
        # Write the header
        writer.writerow(["Dataset", "Class", "Accuracy", "Precision (Macro)", "Recall (Macro)", "F1-Score (Macro)", 
                         "Precision (Weighted)", "Recall (Weighted)", "F1-Score (Weighted)", "Support"])
        
        # Write overall metrics
        writer.writerow([dataset_type, 'Overall', accuracy, precision_macro, recall_macro, f1_macro, 
                         precision_weighted, recall_weighted, f1_weighted, '-'])
        
        # Write per-class metrics
        for label, metrics in class_report.items():
            if label not in ['accuracy', 'macro avg', 'weighted avg']:
                writer.writerow([dataset_type, label, '-', metrics['precision'], metrics['recall'], 
                                 metrics['f1-score'], '-', '-', '-', metrics['support']])

    print(f"\nResults saved to {results_file}")
