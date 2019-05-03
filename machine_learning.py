import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import ComplementNB
from joblib import dump, load
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report


# Machine learning classifier using Naive Bayes method
class Classifier():
    test_size = 0.15  # percentage of training set to use as testing set
    model_file_prefix = 'cnb_model_'
    class_file_prefix = 'cnb_classes_'
    file_suffix = '.joblib'

    def __init__(self):
        self.text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', ComplementNB(norm=True))])  # pipeline of fit/transforms

    def train(self, file, t, print_report=False):
        df = pd.read_csv(file)  # loading training set file
        x = df['native'].dropna()  # drop empty inputs that may have slipped in
        y_names = df['class'].dropna()  # y are currently strings - need to be represented as ints
        self.classes = y_names.unique()  # saving y (target) classes for later
        y = [list(self.classes).index(name) for name in y_names]  # representing y as ints

        #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=self.test_size)  # ideally would split into training and testing sets - but don't due to (small) size of class samples
        x_train = x
        y_train = y
        self.text_clf.fit(x_train, y_train)  # putting training set through the text classification pipeline
        self.save_model_and_classes(t)  # saving the resulting model and its accompanying class names

        accuracy = self.text_clf.score(x_train, y_train)  # get accuracy by comparing model predictions from x_train to ground truth (y_train)
        print('(', file, ') Test set accuracy: ', accuracy)
        if print_report:  # set print_report=True for more information on the model's training performance
            print(classification_report(y_train, self.text_clf.predict(x_train), target_names=self.classes))

    # Print the top class prediction for the input
    def predict(self, new, t, load_model=False):
        if load_model:
            self.text_clf, self.classes = self.load_model_and_classes(t)
        predicted = self.text_clf.predict(new)
        for doc, category in zip(new, predicted):
            print('%r => %s' % (doc, self.classes[category]))

    # Print x ranked class predictions for the input
    def predict_top_x(self, new, x, t, load_model=False):
        if load_model:
            self.text_clf, self.classes = self.load_model_and_classes(t)
        predicted = self.text_clf.predict_proba(new)
        for doc, predictions in zip(new, predicted):
            sorted_pred = np.argsort(-predictions) # stores indexes
            for i in range(x):
                print(i, predictions[sorted_pred[i]], self.classes[sorted_pred[i]])

    # Function to load a model
    def load_model_and_classes(self, t):
        return load(self.model_file_prefix + t + self.file_suffix), load(self.class_file_prefix + t + self.file_suffix)

    # Function to save a model
    def save_model_and_classes(self, t):
        dump(self.text_clf, self.model_file_prefix + t + self.file_suffix)
        dump(self.classes, self.class_file_prefix + t + self.file_suffix)

    # Prints a series of classifier attributes
    def get_attributes(self):
        print("feature_log_prob_:", self.text_clf.named_steps['clf'].feature_log_prob_)
        print("class_count_:", self.text_clf.named_steps['clf'].class_count_)
        print("feature_count_ :", self.text_clf.named_steps['clf'].feature_count_ )
        print("feature_all_ :", self.text_clf.named_steps['clf'].feature_all_ )
