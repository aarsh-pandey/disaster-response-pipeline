# Builtin Libraries
import sys
import sqlite3
import re
import pickle

import warnings
warnings.filterwarnings('ignore')  

# Pandas
import pandas as pd

# NLTK Libraries
import nltk
nltk.download(['punkt','stopwords','wordnet'])

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# sklearn
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.metrics import classification_report,accuracy_score


def load_data(database_filepath):
    '''
    Function to load the Data From SQLite DataBase

    Args:
        database_filepath : path of the database file

    Returns:
        (X,Y,category_names)

        X : features DataFrame
        Y : labels DataFrame
        category_names : List of Column Names which contains different Category

    '''
    # Creating the Connection object of sqlite Database
    conn = sqlite3.connect(database_filepath)

    # getting data from sqlite data base
    df = pd.read_sql('SELECT * FROM messages',conn)

    # Features
    X = df['message'].copy()

    # Labels
    Y = df.drop(columns=['id','message','original','genre']).copy()

    # Category Column names
    category_names = list(Y.columns)

    return X,Y,category_names

def tokenize(text):
    '''
    Function to tokenizing the taxt into words

    Args:
        text : text string which is needed to be tokenize

    Returns:
        tokens : List of word tokens
    '''
    # changing all upper case character to lower case
    text = text.lower()

    # Removing any special character
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    
    tokens = word_tokenize(text)

    # Lemmatizing the text and removing stopwords
    tokens = [WordNetLemmatizer().lemmatize(word, pos='n') for word in tokens \
            if word not in stopwords.words("english")]
    tokens = [WordNetLemmatizer().lemmatize(word, pos='v') for word in tokens]

    # Steming the text
    tokens = [PorterStemmer().stem(word).strip() for word in tokens]

    return tokens


def build_model():
    '''
    Build the model required for training 

    Args:
        None

    Returns:
        
    '''
    # Creating Machine Learning Pipeline
    pipeline = Pipeline([
        ('vect',CountVectorizer(tokenizer = tokenize)),
        ('tfidf',TfidfTransformer()),
        ('clf',MultiOutputClassifier(RandomForestClassifier(min_samples_split=50)) )
    ])


    parameters = {
        'clf__estimator__criterion':['gini','entropy'],
        'clf__estimator__min_samples_split':[10,110],
        'clf__estimator__max_depth':[None,100,500],
        }

    # Creating GridSearch object
    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluate the model based on precision, recall and f1-score

    Args:
        model : object of the model
        X_test : Features for testing
        Y_test : Labels for testing
        category_names : List of Column Names which contains different Category
    '''
    # Predicted value of test data
    Y_test_predicted = model.predict(X_test)

    for i, col in enumerate(category_names):
        print('------------------------')
        print(col)
        print('------------------------')
        print(classification_report(Y_test[col], Y_test_predicted.T[i]))
        print('------------------------')


def save_model(model, model_filepath):
    '''
    Saving the model as pickle file

    Args:
        model : trained classifier object which is needs to be saved
        model_filepath : path of the file where you want to save the model
    '''
    # Dumping the created model to a file
    with open(model_filepath,'wb') as f:
        pickle.dump(model,f)


def main():
    if len(sys.argv) == 3:

        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20)
        
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
