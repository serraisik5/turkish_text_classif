from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack
from sklearn import svm
import numpy as np
import pickle

def vectorize_text(df):
    print("Vectorize text")
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['tokenized_text'])

    with open('./ML_model/vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)

    return tfidf_matrix, tfidf_vectorizer

def combine_features(tfidf_matrix, df):
    encoded_ner_tags_matrix = np.vstack(df['encoded_ner_tags'].values)
    combined_features = hstack([tfidf_matrix, encoded_ner_tags_matrix])
    return combined_features

def train_model(X_train, y_train):
    print("Training..")

    SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    SVM.fit(X_train, y_train)

    return SVM

def save_model(model, file_path):
    with open(file_path, 'wb') as model_file:
        pickle.dump(model, model_file)
