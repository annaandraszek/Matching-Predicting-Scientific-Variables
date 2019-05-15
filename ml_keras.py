from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.models import load_model
from keras.callbacks import EarlyStopping
import tensorflow as tf
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import numpy as np
import pandas as pd


class NeuralNetwork(): #give this arguments like: model type, train/test file
    max_words = 900
    max_len = 10
    tok = Tokenizer(num_words=max_words)
    y_dict = {}
    epochs = 5
    batch_size = 11
    tf.logging.set_verbosity(tf.logging.ERROR)

    def __init__(self, file, model_type='match'):
        df = pd.read_csv(file, dtype={'class': str, 'native': str})
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)
        self.X = df['native']
        Y = df['class']
        le = LabelEncoder()
        Y = le.fit_transform(Y)
        Y = Y.reshape(-1, 1)

        for n in range(len(Y)):
            self.y_dict[int(Y[n])] = df['class'][n]

        self.Y = Y
        self.model = self.RNN()
        self.model.compile(loss='binary_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

    def train(self, model_name):
        X_train, X_test, Y_train, Y_test = train_test_split(self.X, self.Y, test_size=0.15)

        self.tok.fit_on_texts(X_train)
        sequences = self.tok.texts_to_sequences(X_train)
        sequences_matrix = sequence.pad_sequences(sequences, maxlen=self.max_len)
        #y_binary = to_categorical(Y_train)
        self.model.summary()
        self.model.fit(sequences_matrix, Y_train, batch_size=self.batch_size, epochs=self.epochs,
                  validation_split=0.2) #, callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)]

        test_sequences = self.tok.texts_to_sequences(X_test)
        test_sequences_matrix = sequence.pad_sequences(test_sequences, maxlen=self.max_len)

        #Y_test_binary = to_categorical(Y_test)
        accr = self.model.evaluate(test_sequences_matrix, Y_test)
        print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0], accr[1]))
        self.model.save(model_name + '.h5')
        joblib.dump(self.tok, model_name + 'tokeniser.joblib')

    def RNN(self):
        inputs = Input(name='inputs',shape=[self.max_len])
        layer = Embedding(self.max_words,50,input_length=self.max_len)(inputs)
        layer = LSTM(64)(layer)
        layer = Dense(256,name='FC1')(layer)
        layer = Activation('relu')(layer)
        layer = Dropout(0.5)(layer)
        layer = Dense(1, name='out_layer')(layer)
        layer = Activation('sigmoid')(layer)
        model = Model(inputs=inputs,outputs=layer)
        return model


    def load_model_from_file(self, model_name='binary'):
        self.model = load_model(model_name + '.h5')
        self.tok = joblib.load(model_name + 'tokeniser.joblib')
        self.model._make_predict_function()


    def predict(self, strings):
        sequences = self.tok.texts_to_sequences(strings)
        sequences_matrix = sequence.pad_sequences(sequences, maxlen=self.max_len)
        predictions = self.model.predict(sequences_matrix)
        classification = ['unit' if p > 0.5 else 'property'for p in predictions]
        return classification, predictions
