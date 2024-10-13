import pandas as pd
from sklearn.preprocessing import LabelEncoder
from trtokenizer.tr_tokenizer import WordTokenizer
from data_preprocessing import preprocess_text, encode_ner_tags, resample_dataset
from train_SVM import vectorize_text, combine_features, train_model, save_model
from evaluation import evaluate_metrics
from inference import inference, load_model

train_data_path = './data/train.csv'
val_data_path = './data/val.csv'
test_data_path = './data/test.csv'

train_df = pd.read_csv(train_data_path, sep='\t')
val_df = pd.read_csv(val_data_path, sep='\t')
test_df = pd.read_csv(test_data_path, sep='\t')

# Handle Imbalance
train_df = resample_dataset(train_df, target_column='category')

# Preprocess the text data
train_df = preprocess_text(train_df)
val_df = preprocess_text(val_df)
test_df = preprocess_text(test_df)

# Encode NER tags (Use the same OneHotEncoder for all sets)
train_df, ohe_train = encode_ner_tags(train_df)
val_df, _ = encode_ner_tags(val_df, ohe=ohe_train)
test_df, _ = encode_ner_tags(test_df, ohe=ohe_train)

# Vectorize the text data using TF-IDF
tfidf_matrix_train, tfidf_vectorizer = vectorize_text(train_df)
X_train = combine_features(tfidf_matrix_train, train_df)

tfidf_matrix_val = tfidf_vectorizer.transform(val_df['tokenized_text'])  # Use the same vectorizer for validation set
X_val = combine_features(tfidf_matrix_val, val_df)

tfidf_matrix_test = tfidf_vectorizer.transform(test_df['tokenized_text'])  # Use the same vectorizer for test set
X_test = combine_features(tfidf_matrix_test, test_df)

le = LabelEncoder()
y_train = le.fit_transform(train_df['category'])
y_val = le.transform(val_df['category'])
y_test = le.transform(test_df['category'])


# Train the model
SVM = train_model(X_train, y_train)

train_predictions_SVM = SVM.predict(X_train)
val_predictions_SVM = SVM.predict(X_val)
test_predictions_SVM = SVM.predict(X_test)

# Evaluate 
evaluate_metrics(y_train, train_predictions_SVM, "Train")
evaluate_metrics(y_val, val_predictions_SVM, "Validation", results_file="./results/ML_evaluation_val_results.csv")
evaluate_metrics(y_test, test_predictions_SVM, "Test", results_file="./results/ML_evaluation_test_results.csv")

# Save the trained model
save_model(SVM, './models/svm_model.pkl')

