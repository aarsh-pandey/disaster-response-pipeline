
# Disaster Response Pipeline Project

Disaster Response Pipeline to categorize the message sent during any disaster so that message can be sent to an appropriate Disaster Relief Agency

### Table of Contents

1. [Installation](#installation)
2. [QuickStart](#quickstart)
3. [File Descriptions](#files)
4. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

To work with this repository, you just need to clone it to your local machine.

```
$ git clone https://github.com/aarsh-pandey/disaster-response-pipeline.git
```

This project works with Python 3.* , to install dependencies just execute

```
$ pip install -r requirements.txt
```

or you can also install required packages separately by executing

```
$ pip install numpy
$ pip install pandas
$ pip install sklearn
$ pip install plotly
$ pip install nltk
```

if pip doesn't work for you, just try `pip3` instead of `pip`

## QuickStart <a name="quickstart"></a>
- To start the Flask app 
	```
	$ python app/run.py
	```
  now, just type http://0.0.0.0:3001/ to your favourite browser's address bar to see the web app live on your local machine.

- To run ETL pipeline that cleans data and stores in database
	```
	python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
	```
 - To run ML pipeline that trains classifier and saves

	```
	python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
	```

## File Descriptions <a name="files"></a>
<pre>
<code>.
├── <b>README.md</b>
├── <b>app</b> : Flask App Files
│ ├── <b>run.py</b> : Flask file to  run the app
│ └── <b>templates</b>
│ ├── <b>go.html</b>
│ └── <b>master.html</b>
├── <b>data</b> : It contains all ETL Files 
│ ├── <b>DisasterResponse.db</b> :  SQLite DataBase file containing cleaned data after ETL process  
│ ├── <b>disaster_categories.csv</b> :  Disaster Categories CSV file
│ ├── <b>disaster_messages.csv</b> : Messages CSV file
│ └── <b>process_data.py</b> : 
├── <b>models</b> : It contains all ML files
│ ├── <b>classifier.pkl</b> : classifier produced by train_classifier file
│ └── <b>train_classifier.py</b> : ML pipeline classification code
└── <b>requirements.txt</b>
 </code>
</pre>

## Licensing, Authors, Acknowledgements<a name="licensing"></a>
The Data set used in this process is provided by **figure-8** that contains real labeled disaster messages received by an aid organisation during disaster events.
