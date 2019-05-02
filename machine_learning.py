import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
#from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from joblib import dump, load
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report


class Classifier():
    test_size = 0.15

    def __init__(self):
        self.text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', ComplementNB(norm=True))])

    def train(self, file, t):
        df = pd.read_csv(file)
        x = df['native'].dropna()
        y_names = df['class'].dropna() # need to represent this y as ints, and create a symmetric array of the names
        self.classes = y_names.unique()
        y = [list(self.classes).index(name) for name in y_names]

        #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=self.test_size)
        x_train = x
        y_train = y
        self.text_clf.fit(x_train, y_train)
        dump(self.text_clf, 'cnb_model_' + t + '.joblib')
        dump(self.classes, 'cnb_classes_' + t + '.joblib')

        accuracy = self.text_clf.score(x_train, y_train)
        print('(', file, ') Test set accuracy: ', accuracy)
        #print(classification_report(y_train, self.text_clf.predict(x_train), target_names=self.classes))

    def predict(self, new, t, load_model=False):
        if load_model:
            self.text_clf = load('cnb_model.joblib')
            self.classes = load('cnb_classes.joblib')
        predicted = self.text_clf.predict(new)
        for doc, category in zip(new, predicted):
            print('%r => %s' % (doc, self.classes[category]))

    def predict_top_x(self, new, x, t, load_model=False):
        if load_model:
            self.text_clf = load('cnb_model_' + t + '.joblib')
            self.classes = load('cnb_classes_' + t + '.joblib')
        predicted = self.text_clf.predict_proba(new)
        for doc, predictions in zip(new, predicted):
            sorted_pred = np.argsort(-predictions) # stores indexes
            for i in range(x):
                print(i, predictions[sorted_pred[i]], self.classes[sorted_pred[i]])

    def get_attributes(self):
        print("feature_log_prob_:", self.text_clf.named_steps['clf'].feature_log_prob_)
        print("class_count_:", self.text_clf.named_steps['clf'].class_count_)
        print("feature_count_ :", self.text_clf.named_steps['clf'].feature_count_ )
        print("feature_all_ :", self.text_clf.named_steps['clf'].feature_all_ )
