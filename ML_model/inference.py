import pickle
import numpy as np
import pandas as pd
from trtokenizer.tr_tokenizer import WordTokenizer
from train_SVM import combine_features
from evaluation import evaluate_metrics
from sklearn.preprocessing import LabelEncoder

# load the saved model
def load_model(file_path):
    with open(file_path, 'rb') as model_file:
        model = pickle.load(model_file)
    
    return model

def load_ohe(file_path):
    with open(file_path, 'rb') as file:
        ohe = pickle.load(file)
    return ohe

def load_vectorizer(file_path):
    with open(file_path, 'rb') as f:
        vectorizer = pickle.load(f)
    return vectorizer

# preprocess manual input
def preprocess_inference_data(df, word_tokenizer):
    """
    Preprocess the input data by tokenizing the text and encoding the NER tags.
    """
    ohe = load_ohe('./ML_model/ohe.pkl')

    df['text'] = df['text'].str.lower()
    df['tokenized_text'] = df['text'].apply(lambda x: ' '.join(word_tokenizer.tokenize(x)))

    # Encode NER tags using the existing OneHotEncoder
    df['ner_tags'] = df['ner_tags'].apply(lambda x: x.split())
    df['encoded_ner_tags'] = df['ner_tags'].apply(lambda tags: ohe.transform(np.array(tags).reshape(-1, 1)).sum(axis=0))

    
    return df
  

def inference(df, model, word_tokenizer):
    """
    Perform inference on the input dataframe using the trained SVM model.
    """
    # Preprocess the text and encode NER tags
    df = preprocess_inference_data(df, word_tokenizer)

    vectorizer = load_vectorizer('./ML_model/vectorizer.pkl')
    tfidf_matrix = vectorizer.transform(df['tokenized_text'])

    combined_features = combine_features(tfidf_matrix, df)

    predictions = model.predict(combined_features)
    
    return predictions


if __name__ == "__main__":

    svm_model_path = './models/svm_model.pkl'
    svm_model = load_model(svm_model_path)

    df_inference = pd.DataFrame({
        'ner_tags': ['B-business_operation_name O O O O B-business_operation_industry O O B-organization_date_founded O O B-location_containedby O O O O O', 'B-galaxy_name I-galaxy_name O O O O O'],
        'text': ['IP2Location kullanıcılara aynı adla bir yazılım da sunan 2002 yılında kurulmuş Malezya merkezli bir internet şirketidir .',
                'Birçok katalogda sarmal gökada olarak sınıflandırılan NGC 5713, pek çok sarmal gökadadan oldukça farklıdır.']
    })

    # Perform inference
    predictions = inference(df_inference, svm_model, WordTokenizer())
    print(predictions)
