import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

#Loading the model
RNN_model = pickle.load(open('RNN_model.pkl', 'rb'))
tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))
RNN_model.load_weights("weights.h5")

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict_api', methods = ['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    new_data = tokenizer.texts_to_sequences(data.values())
    print(new_data)
    new_data = pad_sequences(new_data, padding = 'post', maxlen = 100)
    new_data= np.array(new_data).reshape((new_data.shape[0],new_data.shape[1], 1))
    prediction = RNN_model.predict(new_data)
    print(prediction)
    print(np.argmax(prediction[0])+1)
    return jsonify(str(np.argmax(prediction[0])+1))


@app.route('/predict', methods = ['POST'])
def predict():
    data = [str(text) for text in request.form.values()]
    # final_input = logreg_transformer.transform(pd.Series(data))
    print(data)
    new_data = tokenizer.texts_to_sequences(data)
    print(new_data)
    new_data = pad_sequences(new_data, padding = 'post', maxlen = 100)
    new_data= np.array(new_data).reshape((new_data.shape[0],new_data.shape[1], 1))
    prediction = RNN_model.predict(new_data)
    print(prediction)
    print(np.argmax(prediction[0])+1)
    prediction = np.argmax(prediction[0])+1
    return render_template("index.html", prediction_text = "The predictive rating for the Review out of a score of 5 is {}".format(prediction))


if __name__ == '__main__':
    app.run(debug=True)