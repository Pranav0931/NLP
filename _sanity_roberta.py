import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import pipeline


def to_bin(label):
    s = str(label).upper()
    return 1 if ('POSITIVE' in s or 'LABEL_1' in s) else 0


df = pd.read_csv('IMDB Dataset.csv')
y = (df['sentiment'] == 'positive').astype(int).values
X = df['review'].values

_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_small = X_test[:250]
y_small = y_test[:250]

model_id = 'textattack/roberta-base-imdb'
clf = pipeline('sentiment-analysis', model=model_id)
preds = []
for t in X_small:
    out = clf(t[:2000], truncation=True, max_length=512)[0]
    preds.append(to_bin(out['label']))

acc = accuracy_score(y_small, preds)
print(f'{model_id}\t{acc:.4f}')
