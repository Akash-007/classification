from tensorflow.keras.models import load_model
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
lema = WordNetLemmatizer()
max_len = 130
rever = {0: 'Real', 1: 'Fake'}
from flask import Flask, request, jsonify, render_template
import pickle
app = Flask(__name__)
tokenizer = pickle.load(open('tokenizer.pickle', 'rb'))
model = load_model('model_spam.h5')


def pre_proces(x):
    x = x.lower()
    x = re.sub('[^a-z0-9]', ' ', x)
    res = []
    for i in x.split(' '):
        if len(i) > 2:
            if i not in stopwords.words('english'):
                res.append(lema.lemmatize(i))
    return ' '.join(res)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    ''' For rendering results on HTML GUI '''
    int_features = [x for x in request.form.values()]
    x = pre_proces(int_features[0])
    seq = tokenizer.texts_to_sequences(x)
    X = pad_sequences(seq, maxlen=max_len)
    mode = load_model('model_spam.h5')
    resi = mode.predict_classes(X)
    output = rever[int(resi[0][0])]
    message = int_features[0]
    return render_template('home.html', prediction_text='The above message was a {} message'.format(output),message=int_features[0])


if __name__ == "__main__":
    app.run(debug=False)