# Summary

* [1. Installation] 
* [2. Project Motivation] 
* [3. File descriptions] 
* [4. How To Interact With The Project]

This is the link to open the website:
https://view6914b2f4-3001.udacity-student-workspaces.com/

# 1. Installation

In order to execute this project, the following libraries were used

* Numpy library
* Pandas library
* Python version 3.8.5
* sqlite3
* sqlalchemy
* nltk
* sklearn
* plotly
* flask

# 2. Project Motivation

This project classifies messages into 36 different categories. It also provides a website to enter a message and it classifies the message depending on what type of message it is and what type of help the person needs. The project applies machine learning algorithms and natural language processing (NLP) to improve the effectiveness of organizations attending people in case of disasters.

# 3. File descriptions

disaster_response_nlp/

├── README.md

├── app/
    templates
    run.py

├── data/
    DisasterResponse.db
    YourDatabaseName.db
    disaster_categories.csv
    disaster_messages.csv
    process_data.py

├── models/
    classifier.plk
    train_classifier.py

├── disaster_response_nlp-model3.ipynb

├── ETL Pipeline Preparation.ipynb

# 4. How to interact with the project

The database used to develop this project contains more than 25,000 messages classified into 36 different categories: 

'related', 'request', 'offer', 'aid_related', 'medical_help',
'medical_products', 'search_and_rescue', 'security', 'military',
'child_alone', 'water', 'food', 'shelter', 'clothing', 'money',
'missing_people', 'refugees', 'death', 'other_aid',
'infrastructure_related', 'transport', 'buildings', 'electricity',
'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
'other_weather', 'direct_report'

The website allows the user to enter a message and it will be classified into the 36 different categories. In order to classify the messages, the website uses a machine learning algorithm: Random Forest Regressor.

The  website also contains visualizations of the data that was used to train the machine learning algorithms used in the project to build the model.

Running this command will allow you to run the ETL pipeline that cleans data and stores in database:
`python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data-DisasterResponse.db`

Running this command will allow you to run the ML pipeline that trains the classifier and saves the model:
`python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

Running the following command in the app's directory will allow you to run the web app.
`python run.py`

This is the link to open the website:
https://view6914b2f4-3001.udacity-student-workspaces.com/


# References

Udacity: Data Scientist Nanodegree

