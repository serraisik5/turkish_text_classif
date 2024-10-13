import pandas as pd
import numpy as np
from trtokenizer.tr_tokenizer import WordTokenizer
from sklearn.preprocessing import OneHotEncoder
import pickle

def load_data(file_path):
    df = pd.read_csv(file_path, sep='\t', encoding='utf-8', header=None)
    df.columns = ['category', 'ner_tags', 'text']

    print(f"Data loaded successfully. Shape of the dataset: {df.shape}")
    return df

def resample_dataset(df, target_column='category', random_state=42):
    """
    Resamples the dataset by applying oversampling to underrepresented classes and undersampling 
    to overrepresented classes.
    """
    category_distribution = df[target_column].value_counts()

    # treshold
    T = int(category_distribution.median())
    
    df_resampled = pd.DataFrame(columns=df.columns)
    
    # Resample each class (oversample or undersample)
    for category, count in category_distribution.items():
        class_subset = df[df[target_column] == category]
        
        if count > T:
            # Undersample
            sampled_data = class_subset.sample(T, replace=False, random_state=random_state)
        elif count < T:
            # Oversample
            sampled_data = class_subset.sample(T, replace=True, random_state=random_state)
        else:
            sampled_data = class_subset
        
        df_resampled = pd.concat([df_resampled, sampled_data])
    df_resampled = df_resampled.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    return df_resampled


def preprocess_text(df):
    print("Preprocessing data...")
    df = df.dropna(subset=['text'])
    df.reset_index(drop=True, inplace=True)
    df['text'] = df['text'].str.lower()

    word_tokenizer = WordTokenizer()
    df['tokenized_text'] = df['text'].apply(lambda x: ' '.join(word_tokenizer.tokenize(x)))
    print("Text tokenization completed")
    return df


def encode_ner_tags(df,ohe=None):
    print("Encoding NER tags...")
    df['ner_tags'] = df['ner_tags'].apply(lambda x: x.split())

    flat_ner_tags = [tag for sublist in df['ner_tags'] for tag in sublist]

    # Fit the OneHotEncoder on the training data NER tags
    if ohe is None:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        ohe.fit(np.array(flat_ner_tags).reshape(-1, 1))
        #save
        with open('./ML_model/ohe.pkl', 'wb') as f:
            pickle.dump(ohe, f)
    
    df['encoded_ner_tags'] = df['ner_tags'].apply(lambda tags: ohe.transform(np.array(tags).reshape(-1, 1)).sum(axis=0))

    print("NER tags encoded")
    return df, ohe
