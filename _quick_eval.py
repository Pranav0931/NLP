import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import pipeline
from sklearn.metrics import accuracy_score


def to_bin(label):
    s = str(label).upper()
    if 'POSITIVE' in s or 'LABEL_1' in s:
        return 1
    return 0


df = pd.read_csv('IMDB Dataset.csv')
y = (df['sentiment'] == 'positive').astype(int).values
X = df['review'].values

X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full)

# quick but representative subset for runtime
n = 3000
Xv = X_val[:n]
yv = y_val[:n]

cands = [
    'textattack/roberta-base-imdb',
    'textattack/bert-base-uncased-imdb',
    'distilbert-base-uncased-finetuned-sst-2-english'
]

for m in cands:
    print('\nMODEL:', m)
    clf = pipeline('sentiment-analysis', model=m)
    preds = []
    for t in Xv:
        out = clf(t, truncation=True, max_length=512)[0]
        preds.append(to_bin(out['label']))
    acc = accuracy_score(yv, preds)
    print('VAL_ACC', round(acc, 4))
