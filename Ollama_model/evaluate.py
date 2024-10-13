from sklearn.metrics import classification_report, accuracy_score

def evaluate_model(test_df):
    y_true = test_df['category_translated'].str.lower()
    y_pred = test_df['predicted_category'].str.lower()

    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)

    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(report)
    return accuracy, report
