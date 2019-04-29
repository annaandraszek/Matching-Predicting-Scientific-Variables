import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.models import load_model
import tensorflow as tf


class Classifier():
    test_size = 0.15

    def __init__(self):
        #self.categories = self.target.unique()
        #self.count_vect = CountVectorizer()
        #self.tfidf_transformer = None
        #self.clf = None
        self.text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', ComplementNB())])

    def train(self, file):
        df = pd.read_csv(file)
        x = df['native'].dropna()
        y_names = df['class'].dropna() # need to represent this y as ints, and create a symmetric array of the names
        self.classes = y_names.unique()
        y = [list(self.classes).index(name) for name in y_names]

        #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=self.test_size)
        x_train = x
        y_train = y
        self.text_clf.fit(x_train, y_train)

        #predicted = self.text_clf.predict(x_test)
        accuracy =  self.text_clf.score(x_train, y_train) # np.mean(predicted == y_test)
        print('Test set accuracy: ', accuracy)

    def predict(self, new):
        predicted = self.text_clf.predict(new)
        for doc, category in zip(new, predicted):
            print('%r => %s' % (doc, self.classes[category]))

    def predict_top_x(self, new, x):
        predicted = self.text_clf.predict_proba(new)
        for doc, predictions in zip(new, predicted):
            sorted_pred = np.argsort(-predictions) # stores indexes
            for i in range(x):
                print(i, predictions[sorted_pred[i]], self.classes[sorted_pred[i]]) #, sorted[i], self.classes[category]))


    # def train(self):
    #     x_train, x_test, y_train, y_test = train_test_split(self.x, self.target, self.test_size)  # may want to do fitting on WHOLE data - due to its nature and size
    #     x_train_counts = self.count_vect.fit_transform(x_train)
    #     self.tfidf_transformer = TfidfTransformer(use_idf=False).fit(x_train_counts)
    #     x_train_tfidf = self.tfidf_transformer.transform(x_train_counts)
    #
    #     self.clf = MultinomialNB().fit(x_train_tfidf, self.target)
    #
    #     predicted = text_clf.predict(docs_test)
    #     np.mean(predicted == twenty_test.target)
    #
    # def predict(self, new_terms):
    #     x_new_counts = self.count_vect.transform(new_terms)
    #     x_new_tfidf = self.tfidf_transformer.transform(x_new_counts)
    #     predicted = self.clf.predict(x_new_tfidf)
    #
    #     for doc, category in zip(new_terms, predicted):
    #         print('%r => %s' % (doc, self.target[category]))

class MachineLearning(): #give this arguments like: model type, train/test file
    max_words = 100
    max_len = 10
    tok = Tokenizer(num_words=max_words)
    y_dict = {}
    epochs = 100
    batch_size = 11
    tf.logging.set_verbosity(tf.logging.ERROR)

    def __init__(self, file, model_type='match'):
        df = pd.read_csv(file)
        self.X = df['native']
        Y = df['class']
        le = LabelEncoder()
        Y = le.fit_transform(Y)
        Y = Y.reshape(-1, 1)

        for n in range(len(Y)):
            self.y_dict[int(Y[n])] = df['class'][n]

        self.Y = Y
        self.model = self.RNN()
        #model.summary()
        self.model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

    def train(self, model_name):
        X_train, X_test, Y_train, Y_test = train_test_split(self.X, self.Y, test_size=0.15)

        self.tok.fit_on_texts(X_train)
        sequences = self.tok.texts_to_sequences(X_train)
        sequences_matrix = sequence.pad_sequences(sequences, maxlen=self.max_len)
        y_binary = to_categorical(Y_train)
        self.model.fit(sequences_matrix, y_binary, batch_size=self.batch_size, epochs=self.epochs,
                  validation_split=0.2)

        test_sequences = self.tok.texts_to_sequences(X_test)
        test_sequences_matrix = sequence.pad_sequences(test_sequences, maxlen=self.max_len)

        Y_test_binary = to_categorical(Y_test)
        accr = self.model.evaluate(test_sequences_matrix, Y_test_binary)
        print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0], accr[1]))
        self.model.save(model_name + '.h5')

    def RNN(self):
        inputs = Input(name='inputs',shape=[self.max_len])
        layer = Embedding(self.max_words,50,input_length=self.max_len)(inputs)
        layer = LSTM(64)(layer)
        layer = Dense(256,name='FC1')(layer)
        layer = Activation('relu')(layer)
        layer = Dropout(0.5)(layer)
        layer = Dense(len(self.y_dict),name='out_layer')(layer)
        layer = Activation('softmax')(layer)
        model = Model(inputs=inputs,outputs=layer)
        return model


    def predict(self, string, model_name):
        model = load_model(model_name + '.h5')
        sequences = self.tok.texts_to_sequences(string)
        sequences_matrix = sequence.pad_sequences(sequences, maxlen=self.max_len)
        predictions = model.predict(sequences_matrix)
        sort_pred = np.argsort(-predictions)
        # print(predictions, sort_pred[:,0], sort_pred[:,1], )

        ranked_predictions = []
        for p in range(sort_pred.shape[1]):
            pred = self.y_dict[int(sort_pred[:, p])]
            ranked_predictions.append(pred)

        return ranked_predictions, model.predict(sequences_matrix)
