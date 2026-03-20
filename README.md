# Movie Review Sentiment Analysis Project

## Aim
Write Python code to perform sentiment analysis using NLP for IMDB movie reviews and classify each review as Positive or Negative.

## Project Objective
Develop a Movie Review Sentiment Analysis system that includes full NLP processing, classical model comparison, strict best model selection with accuracy at least 95 percent, GUI deployment, and GitHub submission readiness.

## Dataset
- IMDB Dataset.csv
- Total records: 50,000 movie reviews
- Labels: positive, negative

## Practical Components Implemented
1. Dataset loading and exploration
2. Text preprocessing
3. Feature extraction using TF IDF and Bag of Words demo
4. Training multiple classification models
5. Performance evaluation and model comparison
6. Best model selection with mandatory threshold
7. GUI implementation

## NLP Pipeline
1. Load dataset and inspect shape, null values, and class distribution
2. Preprocess text with:
    - lowercase conversion
    - HTML removal
    - punctuation and number removal
    - stopword removal
    - lemmatization
3. Split data into train and test sets
4. Vectorize text with TF IDF using ngram range (1, 2)

## Models Compared
- Logistic Regression
- Naive Bayes
- Support Vector Machine using LinearSVC
- Random Forest

## Evaluation Metrics
For each model:
- Accuracy
- Precision
- Recall
- F1 score

The notebook generates a model comparison table and selects the best performer.

## Mandatory Accuracy Rule
The selected best model must have accuracy at least 95 percent.

If all classical models are below 95 percent, the notebook evaluates stronger transformer candidates on the same test set and updates final selection.

If final selected accuracy is still below 95 percent, execution stops with an error to enforce the requirement.

## Best Model Output
The notebook clearly prints:
- Best model name
- Final accuracy
- Pass status for the at least 95 percent requirement

## Saved Artifacts
The notebook saves:
- best_model.pkl
- vectorizer.pkl
- model_meta.pkl

## GUI Implementation
Framework used:
- Gradio

User flow:
1. Enter a movie review
2. Click predict
3. View predicted sentiment as Positive or Negative

The GUI always uses the final selected best model.

## Example Predictions
The notebook includes sample review predictions in a dedicated section after GUI setup.

## How To Run
1. Place IMDB Dataset.csv in the same folder as the notebook
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run notebook:

```bash
jupyter notebook Movie_Review_Sentiment_Analysis_Project.ipynb
```

4. Execute all cells from top to bottom

## Project Files
- Movie_Review_Sentiment_Analysis_Project.ipynb
- IMDB Dataset.csv
- requirements.txt
- README.md
- best_model.pkl
- vectorizer.pkl
- model_meta.pkl

## Final Report Content Guide
Include the following sections in your PDF report:

1. Objective
    - Build sentiment analysis for movie reviews using NLP and machine learning

2. Methodology
    - Data loading and exploration
    - Text preprocessing
    - TF IDF feature extraction with ngram (1,2)
    - Model training and evaluation
    - Best model selection with threshold enforcement

3. Models Used
    - Logistic Regression
    - Naive Bayes
    - LinearSVC
    - Random Forest
    - Optional stronger fallback models if needed to satisfy threshold

4. Results and Comparison
    - Accuracy, Precision, Recall, F1 score table
    - Best model name and final accuracy
    - Evidence that final selected model is at least 95 percent

5. Conclusion
    - Final selected model and deployment readiness
    - GUI functionality and practical usability
