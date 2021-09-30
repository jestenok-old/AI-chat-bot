import random
import json
import pickle
import numpy as np
import re
import nltk
import pymorphy2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
from nltk.corpus import stopwords
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

morph = pymorphy2.MorphAnalyzer()

intents = json.loads(open('intents.json', encoding='utf-8').read())

words = []
classes = []
documents = []
# ignore_letters = ['?', '!', '.', ',', '/', '-', '(', ')']
# ignore_letters.extend(stopwords.words('russian'))
for intent in intents:
    word_list = nltk.word_tokenize(intent['question'], language='russian')
    words.extend(word_list)
    documents.append((word_list, intent['question']))
    if intent['question'] not in classes:
        classes.append(intent['question'])

print('Очко пройдено')

words = set(words)
words = [morph.parse(re.sub(r'[^а-я]', '', word))[0].normal_form for word in words]
words = sorted(set(words))
classes = sorted(set(classes))

pickle.dump(words, open('model/words.pkl', 'wb'))
pickle.dump(classes, open('model/classes.pkl', 'wb'))

training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [morph.parse(word.lower())[0].normal_form for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training, dtype=list)

train_x = list(training[:, 0])
train_y = list(training[:, 1])

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

hist = model.fit(np.array(train_x), np.array(train_y), epochs=800, batch_size=15, verbose=1)
model.save('model/chatbotmodel.h5', hist)
