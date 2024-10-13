import pandas as pd
from deep_translator import GoogleTranslator
from sklearn.model_selection import train_test_split

def load_and_process_data(filepath):
    df = pd.read_csv(filepath, sep='\t', encoding='utf-8', header=None)
    df.columns = ['category', 'ner_tags', 'text']
    _, df = train_test_split(df, test_size=50/len(df),random_state=42)

    unique_categories = df['category'].unique()
    print("\nTranslating Categories")
    category_translation_dict = translate_categories(unique_categories)
    print("Translated")
    df['category_translated'] = df['category'].map(category_translation_dict)
    df['category_translated'] = df['category_translated'].str.replace('TV', 'tv')
    return df

def translate_categories(categories):
    category_translation_dict = {}
    for category in categories:
        translated_category = GoogleTranslator(source='en', target='tr').translate(category)
        category_translation_dict[category] = translated_category
    return category_translation_dict
