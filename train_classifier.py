import sys
# import libraries
import sqlite3
from sqlalchemy import create_engine
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pickle

import numpy as np
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])

def load_data(database_filepath):
    """
    input: database directory
    
    output: 
    X - dataframe with the messages
    Y - Categories of each message
    list(df.columns[4:]) - names of the categories
    """
    
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('YourTableName', engine)
    
    # defining X and Y to then train the model
    X = df['message']
    Y = df[df.columns[4:]]
    
    return X, Y, list(df.columns[4:])


def tokenize(text):
    """input: text - the text messages of the database
    word_tokenize - process the text and separate each word as a list's item
    stopwords - remove english stopwords like: a, an, and, to, etc...
    lemmatizer - normalize the words
    lower() - lowercase all words
    strip() - remove whitespaces
    output: a list with each transformed and procesed word
    """
    
    # tokenize (vectorize) and remove stopwords
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stopwords.words("english")]
    
    # defining the lemmatizer, lemmatize, lowercase, and removing whitespaces
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    """
    this function build the ML model to process the text
    pipeline - simplifies the text's processing
    CountVectorizer - vectorize the words of each message (text)
    tfidf - reflects how important is a word to the message
    clf - it's the ML classifier
    MultiOutputClassifier - It's used because we have 36 categories
    RandomForestClassifier - the classifier
    """
    
    # defining the pipeline to process the text and to define the classifier

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
#         ('clf', MultiOutputClassifier(KNeighborsClassifier()))
        
    ])
    
    """
    GridSearchCV iterate through the parameters, in this case, the 
    number of estimators of the RandomForest classifier to assess
    which model's parameter is the most accurate
    """
    
    # more estimators would create a more accurate model
    # but it gets too heavy and takes long time to run
    parameters = {
    
#     'clf__estimator__n_neighbors': (5, 10)
        'clf__estimator__n_estimators': [1, 3]
#         'clf__estimator__bootstrap': [True, False]
    
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv 
    

def evaluate_model(model, X_test, Y_test, category_names):
    test = np.array(Y_test)
    pred = model.predict(X_test)
    col_names = category_names
    """    
    inputs - 
    test - y_test data.
    pred - predicted data.
    col_names - categories' names.
       
    output - 
    df_scores: accuracy, precision, recall and f1 score for each
    category.
    """
    
    scores = []
    
    # accuracy, precision, recall and f1_score for each category
    for i in range(len(col_names)):
        accuracy = accuracy_score(test[:, i], pred[:, i])
        precision = precision_score(test[:, i], pred[:, i])
        recall = recall_score(test[:, i], pred[:, i])
        f1_sco = f1_score(test[:, i], pred[:, i])
        
        scores.append([accuracy, precision, recall, f1_sco])
    
    scores = np.array(scores)
    df_scores = pd.DataFrame(
        data = scores, index = col_names, columns = [
            'Accuracy', 'Precision', 'Recall', 'F1_score'])
      
    print(df_scores)   


def save_model(model, model_filepath):
    # Exporting the model as a pickle file
    pickle.dump(model, open(model_filepath, 'wb'))
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()