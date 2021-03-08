import json
import plotly
import pandas as pd
import re

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify

from plotly.graph_objs import Bar

from sqlalchemy import create_engine

import pickle


app = Flask(__name__)



def tokenize(text):
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



# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Messages', engine)

# load model
with open('../models/classifier.pkl','rb') as f:
    model = pickle.load(f)


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    category_related_counts = df.groupby('category_related').count()['message']
    category_related_names = ['yes' if i==1 else 'No' for i in list(category_related_counts.index)]


    requests_counts = df.groupby(['category_related','category_request']).count().loc[1,'message']
    category_requests_names = ['yes' if i==1 else 'No' for i in list(requests_counts.index)]

    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=category_related_names,
                    y=category_related_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Messages Related with Disaster',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category Related"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=category_requests_names,
                    y=requests_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message which were Requests <br> out of Related Messages',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Requests"
                }
            }
        }
        
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
