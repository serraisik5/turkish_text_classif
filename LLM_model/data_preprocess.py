# preprocess.py
import pandas as pd
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
import json

def load_datasets():
    """
    Loads the train, test, and val datasets, applies label encoding, and returns the datasets with the label mapping.
    """
    train_df = pd.read_csv('./data/train.csv', sep='\t')
    test_df = pd.read_csv('./data/test.csv', sep='\t')
    val_df = pd.read_csv('./data/val.csv', sep='\t')

    train_df.columns = ['category', 'ner_tags', 'text']
    test_df.columns = ['category', 'ner_tags', 'text']
    val_df.columns = ['category', 'ner_tags', 'text']

    # Encode the labels
    le = LabelEncoder()
    train_df['label'] = le.fit_transform(train_df['category'])
    test_df['label'] = le.transform(test_df['category'])
    val_df['label'] = le.transform(val_df['category'])

    special_tokens = create_tokens_from_NER_tags(train_df, test_df, val_df)

    train_df['annotated_text'] = train_df.apply(annotate_text, axis=1,args=(special_tokens,))
    test_df['annotated_text'] = test_df.apply(annotate_text, axis=1,args=(special_tokens,))
    val_df['annotated_text'] = val_df.apply(annotate_text, axis=1,args=(special_tokens,))

    # HuggingFace Datasets
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    val_dataset = Dataset.from_pandas(val_df)

    label_mapping = dict(enumerate(le.classes_))
    print(f"\nUnique label count: {len(label_mapping)}")
    print(f"\nLabel Mapping: {label_mapping}\n")

    with open("./data/label_mapping.json", "w") as f:
        json.dump(label_mapping, f)

    return train_dataset, test_dataset, val_dataset, label_mapping

def create_tokens_from_NER_tags(train_df, test_df, val_df):
    unique_ner_tags = set(
    tag
    for tags in pd.concat([train_df['ner_tags'], test_df['ner_tags'], val_df['ner_tags']])
    for tag in tags.split()
    )
    unique_ner_tags.discard('O')  

    # Create special tokens for each unique NER tag
    special_tokens = {tag: f"[{tag.upper()}_START]" for tag in unique_ner_tags}
    for tag in unique_ner_tags:
        if tag.startswith('B-'):
            i_tag = tag.replace('B-', 'I-')
            special_tokens[i_tag] = f"[{i_tag.upper()}_CONT]"

    return special_tokens

def annotate_text(row,special_tokens):
    words = row['text'].split()
    tags = row['ner_tags'].split()
    annotated_text = []
    for word, tag in zip(words, tags):
        if tag != 'O':
            # Add special tokens before and after the entity
            annotated_word = f"{special_tokens[tag]} {word} {special_tokens[tag]}"
            annotated_text.append(annotated_word)
        else:
            annotated_text.append(word)
    return ' '.join(annotated_text)

def preprocess_function(dataset, tokenizer):
    """
    Preprocess the dataset using the tokenizer: tokenizes the text, removes unnecessary columns,
    and returns the tokenized dataset.
    """
    def tokenize(examples):
        inputs = tokenizer(examples['annotated_text'], truncation=True, padding='max_length', max_length=512)
        inputs.pop("token_type_ids", None)
        return inputs

    dataset = dataset.remove_columns(['ner_tags', 'category', 'text'])
    tokenized_dataset = dataset.map(tokenize, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(["annotated_text"])  # Remove text as model only accepts numbers

    return tokenized_dataset

def tokenize_datasets(train_dataset, test_dataset, val_dataset, tokenizer):
    tokenized_ds = {}
    splits = {"train": train_dataset, "test": test_dataset, "val": val_dataset}
    
    for split, dataset in splits.items():
        tokenized_ds[split] = preprocess_function(dataset, tokenizer)
    
    print("Current label count in train dataset:", len(set(tokenized_ds["train"]["label"])))
    print("Current label count in test dataset:", len(set(tokenized_ds["test"]["label"])))

    return tokenized_ds