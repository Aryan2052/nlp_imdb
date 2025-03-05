# Sentiment Analysis using Machine Learning

This project performs sentiment analysis on IMDB movie reviews using Natural Language Processing (NLP) and Machine Learning models. The goal is to classify movie reviews as either positive or negative.

## Features
- Preprocesses text data by cleaning, tokenizing, and lemmatizing reviews.
- Uses three classification models: Logistic Regression, Naive Bayes, and Support Vector Machine (SVM).
- Implements TF-IDF vectorization for feature extraction.
- Evaluates model performance using accuracy, precision, recall, and F1-score.
- Performs cross-validation for robust evaluation.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/sentiment-analysis.git
   cd sentiment-analysis
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download necessary NLTK resources:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   nltk.download('averaged_perceptron_tagger')
   ```

## Dataset
- The dataset consists of IMDB movie reviews with labeled sentiments (`positive` or `negative`).
- The data is loaded using Pandas and preprocessed for machine learning models.

## Project Workflow
1. **Import Dependencies** - Load required libraries for data processing, NLP, and machine learning.
2. **Download NLTK Resources** - Ensure required linguistic datasets are available.
3. **Load Dataset** - Read and explore the dataset.
4. **Preprocess Text** - Clean and transform the text data.
5. **Prepare Data for ML** - Encode labels and split into training & testing sets.
6. **Train Models** - Train Logistic Regression, Naive Bayes, and SVM classifiers.
7. **Evaluate Performance** - Use classification reports and confusion matrices.
8. **Cross-Validation** - Ensure model robustness using Stratified K-Fold cross-validation.
9. **Make Predictions** - Predict sentiment of new reviews using the best-performing model.

## Model Performance
| Model | Accuracy |
|--------|------------|
| Logistic Regression | 88% |
| Naive Bayes | 85% |
| Support Vector Machine | 88% |

## Usage
To train and evaluate the models, run:
```python
python main.py
```
To predict sentiment for a custom review, use:
```python
predict_sentiment("The movie was fantastic and thrilling!")
```

## Future Improvements
- Implement deep learning models such as LSTMs or Transformers.
- Enhance feature extraction with word embeddings.
- Deploy as a web application for real-time sentiment analysis.

## License
This project is open-source and available under the MIT License.

