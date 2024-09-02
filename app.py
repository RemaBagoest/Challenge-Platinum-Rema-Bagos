
'''
Alur API
Menerima data .csv, cleaning tweet column, tokenisasi feature, tf-idf atau 
bag of word, proses oleh model dan munculkan dengan .json_response 
'''
import re
import pandas as pd
import numpy as np
from flask import Flask,jsonify, request,send_file
import io
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from flask import request
from flasgger import Swagger, LazyString, LazyJSONEncoder
from flasgger import swag_from
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.sparse import hstack
from sklearn.preprocessing import StandardScaler
import pickle
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense,Embedding, LSTM, SpatialDropout1D, SimpleRNN, Activation
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.layers import Flatten
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)
app.json_encoder = LazyJSONEncoder
swagger_template = {
      "info": {
        "title": "API untuk Prediksi Sentiment pada Kolom Komentar Twitter di Indonesia",
        "version": "1.0.0",
        "description": "API for Sentiment Analysis from Twitter Comment with LSTM and TensorFlow",
        "description": "Kelompok 02 Binar Academy Bootcamp Wave 22",
        "description" : "Anggota Kelompok: Rema Bagos Pudyastowo, Angga Prasetyo, Eko Hardani Wibowo, dan Yusuf Cahyo Triputra"
    },
    "host": "127.0.0.1:5000"  
}

swagger_config = {
      "headers" : [],
      "specs" : [
            {
                  "endpoint" : 'docs', 
                  "route" : '/docs.json'
            }
      ],
      "static_url_path" : '/flasgger_static',
	"swagger_ui" : True,
	"specs_route" : "/docs/"
}
swagger = Swagger(app, template=swagger_template, config = swagger_config)
dict_kamusalay = pd.read_csv('data_add/new_kamusalay.csv', encoding = 'latin-1', header = None)
dict_kamusalay = dict_kamusalay.rename(columns={0:'original', 1:'replacement'})
dict_abusive = pd.read_csv('data_add/abusive.csv', encoding='latin-1', header = None)
dict_abusive = dict_abusive.rename(columns={0:'abusive'})
id_stopword_dict = pd.read_csv('../rema_challenge_platinum/data_add/stopwordbahasa_nya.csv', header = None)
id_stopword_dict = id_stopword_dict.rename(columns = {0 : 'stopword'})


def lowercase(text):
      return text.lower()

def remove_hex_sequences(text):
    hex_pattern = re.compile(r'(\\x[0-9a-fA-F]{2})+')
    return hex_pattern.sub('', text)

def clean_text(text):
    text = re.sub(r'\\x[0-9a-fA-F]{2}', '', text)
    text = re.sub(r'\\[a-zA-Z0-9]{1,}', '', text)
    return text

def remove_unnecessary_char(text):
    text = re.sub('\n',' ',text) 
    text = re.sub('rt',' ',text) 
    text = re.sub('user',' ',text) 
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))',' ',text) 
    pattern = re.compile(r'\\x[0-9A-Fa-f]{2}')
    text = pattern.sub(' ', text)
    text = re.sub('  +', ' ', text) 
    return text
def remove_nonaplhanumeric(text):
    text = re.sub('[^0-9a-zA-Z]+', ' ', text) 
    return text
kamusalay_map = dict(zip(dict_kamusalay['original'], dict_kamusalay['replacement']))
def normalize_alay(text):
    return ' '.join([kamusalay_map[word] if word in kamusalay_map else word for word in text.split(' ')])
def stopwords(text):
      text = ' '.join([' ' if word in id_stopword_dict.stopword.values else word for word in text.split(' ')])
      text = re.sub(' +', ' ', text)
      text = text.strip()
      return text

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
factory = StemmerFactory()
stemmer = factory.create_stemmer()


def stemming(text):
    return stemmer.stem(text)


def preprocess(text):
    text = lowercase(text) 
    text = clean_text(text)
    text = remove_hex_sequences(text)
    text = remove_nonaplhanumeric(text) 
    text = remove_unnecessary_char(text) 
    text = normalize_alay(text) 
    text = stemming(text)
    text = stopwords(text) 
    return text

# Mendefinisikan model LSTM 
max_features = 100000
tokenizer = Tokenizer(num_words=max_features, split=' ', lower=True)

# Membuka x tokenizer untuk model LSTM 
file = open("./feature_lstm/X_LSTM_pad_sequences_2.pickle", 'rb')
feature_file_from_lstm_x = pickle.load(file)
file.close()

# Membuka y tokenizer untuk model LSTM
file = open("./feature_lstm/Y_LSTM_labels_2.pickle", 'rb')
feature_file_from_lstm_y = pickle.load(file)
file.close()

import joblib
# load model lstm
model_file_from_lstm = load_model('./model_lstm/lstm_model_tanpa_nya.h5')

# load model mlp
model_file_from_mlp = joblib.load('./model_mlp/model_mlp_3.pickle', 'rb')
sentiment = ['negative', 'neutral', 'positive']
vectorizer = joblib.load('./feature_mlp/vectorizer_mlp.pickle', 'rb') 


# Model MLP
file = open('./feature_mlp/feature_mlp.pickle', 'rb')
feature_file_from_mlp = pickle.load(file)
file.close()


@swag_from("D:/Rems Tech Agency/rema_challenge_platinum/docs/hello_world.yml", methods = ['GET'])
@app.route('/', methods = ['GET']) # hanya mengakses data tanpa memberikan inputan
def hello_world():
      json_response = {
        'status_code' : 200, 
             'description': "Welcome to API for SentimentAnalysis !!",
            'data': "To continue please access the '/text-processing' endpoint for data cleaning via form, '/text-processing-file' for data cleaning via file upload or '/docs' to see all available menus.",    
      }

      response_data = jsonify(json_response)
      return response_data

## endpoint form LSTM 
@swag_from("D:/Rems Tech Agency/rema_challenge_platinum/docs/lstm_form.yml", methods=['POST'])
@app.route('/lstm_form',  methods = ['POST'])
def lstm_form():

      original_text_lstm_form = request.form.get('text')
      text = [preprocess(original_text_lstm_form)]
      feature_lstm_form = tokenizer.texts_to_sequences(text)
      feature_lstm_form = pad_sequences(feature_file_from_lstm_x, maxlen= feature_file_from_lstm_x.shape[1])
      prediction_lstm_form = model_file_from_lstm.predict(feature_lstm_form)
      get_sentiment_lstm_form = sentiment[np.argmax(prediction_lstm_form[0])]

      json_response = {
            'status_code':  200, 
            'description': 'Hasil Prediksi Sentiment dengan Model LSTM melalui Form', 
            'data' : {
               'text' : original_text_lstm_form,
               'sentiment' : get_sentiment_lstm_form
            },
      } 

      response_data = jsonify(json_response)
      return response_data


# Upload file LSTM
@swag_from("D:/Rems Tech Agency/rema_challenge_platinum/docs/lstm_file.yml",methods=['POST'])
@app.route('/lstm_file', methods = ['POST'])

def lstm_file():
    
    original_text_lstm_file = request.files.getlist('file')[0] 
    df = pd.read_csv(original_text_lstm_file, encoding = 'latin-1')
    texts = df['Tweet'].to_list()
    text = [preprocess(text) for text in texts]
    feature_lstm_form = tokenizer.texts_to_sequences(text)
    feature_lstm_form = pad_sequences(feature_file_from_lstm_x, maxlen= feature_file_from_lstm_x.shape[1])
    prediction_lstm_form = model_file_from_lstm.predict(feature_lstm_form)

    data = [
        {'text': text, 'sentiment': sentiment[np.argmax(prediction_lstm_form)]}
        for text, prediction_lstm_form in zip(texts, prediction_lstm_form)
    ]

    json_response = {
        'status_code': 200,
        'description': 'Hasil Prediksi Sentiment dengan Model LSTM melalui Form',
        'data': data
    }

    response_data = jsonify(json_response)
    return response_data

## endpoint form TensorFlow 
@swag_from("D:/Rems Tech Agency/rema_challenge_platinum/docs/mlp_form.yml", methods=['POST'])
@app.route('/mlp_form',  methods = ['POST'])
def mlp_form():

    original_text_mlp_form = request.form.get('text')
    text = feature_file_from_mlp.transform([preprocess(original_text_mlp_form)])
    results_mlp = model_file_from_mlp.predict(text)[0]

    json_response = {
        'status_code':  200, 
        'description': 'Hasil Prediksi Sentiment dengan Model MLP Classifier melalui Form', 
        'data' : {
            'text' : original_text_mlp_form,
            'sentiment' : results_mlp
            },
    } 

    response_data = jsonify(json_response)
    return response_data
# Upload file LSTM

@swag_from("D:/Rems Tech Agency/rema_challenge_platinum/docs/mlp_file.yml",methods=['POST'])
@app.route('/mlp_file', methods = ['POST'])
def mlp_file():
    
    original_text_mlp_file = request.files.getlist('file')[0] # 
    # Import file csv ke Pandas
    df = pd.read_csv(original_text_mlp_file, encoding = 'latin-1')
    # Ambil teks yang akan diproses dalam format list
    texts = df['Tweet'].to_list()
    text = feature_file_from_mlp.transform([preprocess(text) for text in texts])
    results_mlp = model_file_from_mlp.predict(text)
      # Prepare the data for the JSON response
    data = [
        {'text': text, 'sentiment': sentiment}
        for text, sentiment in zip(texts, results_mlp)
    ]

    json_response = {
        'status_code': 200,
        'description': 'Hasil Prediksi Sentiment dengan Model LSTM melalui Form',
        'data': data
    }
   
    response_data = jsonify(json_response)
    return response_data

if __name__ == '__main__':
    app.run() 
