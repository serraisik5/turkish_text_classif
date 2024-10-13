from sklearn.model_selection import train_test_split
from data_preprocessing import load_and_process_data
from model import get_classification_chain, classify_text

def train_model(filepath):
    df = load_and_process_data(filepath)
    labels = ", ".join(sorted(set(df['category_translated'])))
    chain = get_classification_chain(labels)

    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
    test_df['predicted_category'] = test_df['text'].apply(lambda text: classify_text(chain, text, labels))

    return test_df
