import pickle
from tensorflow.keras.models import load_model
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
lema = WordNetLemmatizer()
max_len = 130
rever = {0: 'spam', 1: 'ham'}
from flask import Flask, request, jsonify, render_template
import pickle
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=False)