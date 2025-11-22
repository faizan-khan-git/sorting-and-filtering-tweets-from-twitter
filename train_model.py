import pandas as pd
import numpy as np
import os
import joblib
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# --- CONFIGURATION ---
DATASET_DIR = 'TwitterDatasets'
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

def clean_text(text):
    """
    Clean text but KEEP important sentiment indicators.
    """
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'\n', ' ', text)
    return text

def train_and_save(name, df, target_col='label'):
    print(f"\n--- Training {name} Model (Random Forest) ---")
    
    # 1. Preprocessing
    print(f"Cleaning {len(df)} tweets...")
    df['text'] = df['text'].apply(clean_text)
    
    X = df['text']
    y = df[target_col]

    # 2. Splitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # 3. Pipeline Configuration (Random Forest)
    # - n_estimators=100: Builds 100 separate decision trees.
    # - n_jobs=-1: Uses ALL cores on your M2 Mac for speed.
    # - max_depth=None: Allows trees to grow fully (High Training Accuracy).
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 3),      # Tri-grams for context
            max_features=15000,      # Reduced slightly for RF speed, but still high
            sublinear_tf=True
        )),
        ('clf', RandomForestClassifier(
            n_estimators=100, 
            n_jobs=-1, 
            random_state=42
        )) 
    ])

    # 4. Training
    print("Fitting Random Forest ")
    pipeline.fit(X_train, y_train)

    # 5. Evaluation
    train_acc = pipeline.score(X_train, y_train) * 100
    test_acc = pipeline.score(X_test, y_test) * 100
    
    print(f"✅ {name} TRAINING Accuracy: {train_acc:.2f}%")
    print(f"✅ {name} TESTING  Accuracy: {test_acc:.2f}%")
    
    # Save
    filename = os.path.join(MODEL_DIR, f'{name.lower()}_model.pkl')
    joblib.dump(pipeline, filename)
    print(f"Saved to {filename}")

def load_data():
    # FILTER DATA (Hate/Offensive)
    dfs_filter = []
    for file in ['hate_train.csv', 'offensive_train.csv', 'hate.csv', 'offensive.csv']:
        path = os.path.join(DATASET_DIR, file)
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                if 'text' not in df.columns: 
                    df.rename(columns={df.columns[0]: 'text', df.columns[1]: 'label'}, inplace=True)
                df = df[['text', 'label']]
                dfs_filter.append(df)
                print(f"Loaded {file}")
            except Exception as e:
                print(f"Skipped {file}: {e}")
            
    if dfs_filter:
        full_filter_df = pd.concat(dfs_filter, ignore_index=True)
        full_filter_df['label'] = full_filter_df['label'].astype(str)
        train_and_save("Filter", full_filter_df)
    else:
        print("❌ CRITICAL: Could not find any Hate/Offensive CSV files.")

    # SORT DATA (Sentiment)
    path_sent = os.path.join(DATASET_DIR, 'sentiment_train.csv')
    if not os.path.exists(path_sent): path_sent = os.path.join(DATASET_DIR, 'sentiment.csv')

    if os.path.exists(path_sent):
        try:
            df_sent = pd.read_csv(path_sent)
            if 'text' not in df_sent.columns: 
                df_sent.rename(columns={df_sent.columns[0]: 'text', df_sent.columns[1]: 'label'}, inplace=True)
            df_sent = df_sent[['text', 'label']]
            print(f"Loaded sentiment data")
            df_sent['label'] = df_sent['label'].astype(str)
            train_and_save("Sort", df_sent)
        except Exception as e: print(f"Error loading sentiment: {e}")
    else:
        print("❌ CRITICAL: Could not find sentiment_train.csv")

if __name__ == "__main__":
    print("🚀 Starting Random Forest ")
    load_data()
    print("\nDONE! Models updated.")