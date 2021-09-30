import random
import json
import pickle
import numpy as np
import nltk
import pymorphy2
from tensorflow.keras.models import load_model
from nltk.corpus import stopwords

morph = pymorphy2.MorphAnalyzer()
intents = json.loads(open('intents.json', encoding='utf-8').read())

words = pickle.load(open('model/words.pkl', 'rb'))
classes = pickle.load(open('model/classes.pkl', 'rb'))
model = load_model('model/chatbotmodel.h5')

ignore_letters = ['?', '!', '.', ',', '/', '-', '(', ')']
# ignore_letters.extend(stopwords.words('russian'))

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [morph.parse(word.lower())[0].normal_form for word in sentence_words if word not in ignore_letters]
    return sentence_words


def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_TRESHOLD = 0.01
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_TRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list


def get_response(message):
    ints = predict_class(message)
    result = "Не найдено соответствий"
    question = ints[0]['intent']
    for l in intents:
        if l['question'] == question:
            result = random.choice(l['answers'])
    return result
