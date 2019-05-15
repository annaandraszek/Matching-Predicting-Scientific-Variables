import preprocessing
import resource_creation
from ml_keras import NeuralNetwork
from machine_learning import Classifier
import pandas as pd
import numpy as np

# Raw downloaded/created datasets which will have features (properties/units) extracted and used in the training set
datasets = ['AIMS_NingalooReef_AirTemperature_WindSpeed(Scalaravg10min)_WindDirection(VectorA-edited.csv',
               'ELI01m10m_201901.csv', 'IDCJDW4019.201903.csv', 'undownloadable.csv', ]

#Downloaded datasets in which the native features can also seve as their class features
raw_qudt_datasets = ['qudt-property.csv', 'qudt-unit.csv']

#Pre-processed versions of the qudt datasets
only_qudt_datasets = ['proc_qudt-property.csv', 'proc_qudt-unit.csv']

unit_vocab = {str.lower(t) for t in resource_creation.create_reference('my_unit.csv', raw_file=False)}
property_vocab = {str.lower(t) for t in resource_creation.create_reference('my_property.csv', raw_file=False)}


def nn_train(file):
    model = 'binary'
    ml = NeuralNetwork(file)
    ml.train(model)


def nn_train_predict(file):
    model = 'binary'
    ml = NeuralNetwork(file)
    ml.train(model)
    test_xs = ['wind', 'degree', 'horsepower', 'hp water', 'foot', 'concentration']
    print(ml.predict(test_xs, model))


def nn_just_predict(string):
    model = 'binary'
    ml = NeuralNetwork('property_or_unit.csv')
    return ml.predict(string, model, load_from_file=True)


def app_binary_predict(string, model):
    return model.predict(string)


def app_class_predict(model, string, t):
    return model.predict_top_x([string], t)


def app_load_models():
    binary_model = NeuralNetwork('property_or_unit.csv')
    binary_model.load_model_from_file()
    properties_model = Classifier()
    properties_model.load_model_and_classes('p')
    units_model = Classifier()
    units_model.load_model_and_classes('u')
    return binary_model, properties_model, units_model


#Cleans up/pre-processes the raw qudt datasets and saves as files in only_qudt_datasets
def process_raw_qudt():
    for dataset in raw_qudt_datasets:
        resource_creation.create_reference(dataset, raw_file=True)


def segment_user_string(string, binary_m):
    window_size = 1
    words = string.split(sep=" ")
    num_segments = len(words) - (window_size-1)
    segments = []

    if len(words) <= window_size:
        segments = [words]
    else:
        for i in range(num_segments):
            segment = [words[n] for n in range(i, i + window_size)]
            segments.append(segment)
    print(segments)
    labels, probabilities = app_binary_predict(segments, binary_m)
    print(labels, probabilities)
    property_words = []
    unit_words = []
    for i in range(len(labels)):
        print(i, segments[i])
        if labels[i] == 'property':
            if len(segments) == 1:
                property_words.extend(segments[i])
            else:
                #if property_words:
                #    print(segments[i][0], property_words[-1])
                if property_words and segments[i][0] == property_words[-1]:
                    property_words.extend(segments[i][1:])
                else:
                    property_words.extend(segments[i])
        if labels[i] == 'unit':
            if len(segments) == 1:
                unit_words.extend(segments[i])
            else:
                if unit_words and segments[i][0] == unit_words[-1]:
                    unit_words.extend(segments[i][1:])
                else:
                    unit_words.extend(segments[i])
    print(property_words, unit_words)
    return " ".join(property_words), " ".join(unit_words)


def app_process_user_input(s, binary_m):
    s_tokens = set(s.split())
    dictionary_s_tokens = (s_tokens.intersection(unit_vocab)).union(s_tokens.intersection(property_vocab))
    property_and_unit_tokens = unit_vocab.union(property_vocab)
    if len(dictionary_s_tokens) != len(s_tokens):
        s = str.lower(s)
        # property_and_unit_tokens = [str.lower(token) for token in property_and_unit_tokens]
        s = preprocessing.solve_abbreviations(s, property_and_unit_tokens,
                                              input_is_string=True)  # need to adapt this and next method for just strings
        s = preprocessing.solve_similar_spelling(s, property_and_unit_tokens, input_is_string=True)
        s_tokens = set(s.split())

    user_property, user_unit = segment_user_string(s, binary_m)
    return user_property, user_unit


def app_user_input(s, binary_m, property_m, unit_m):
    output = app_process_user_input(s, binary_m)
    if output == 1:
        return
    else:
        properties, units = output
        if len(properties) > 0:
            property_predictions = app_run_classifier(properties, 'p', property_m, ranked=True)
            return property_predictions
        if len(units) > 0:
            unit_predictions = app_run_classifier(units, 'u', unit_m, ranked=True)
            return unit_predictions


def process_user_input():
    s = str(input('Enter a string to predict:'))
    if s == 'xxx':
        return 1
    s_tokens = set(s.split())
    dictionary_s_tokens = (s_tokens.intersection(unit_vocab)).union(s_tokens.intersection(property_vocab))
    property_and_unit_tokens = unit_vocab.union(property_vocab)
    if len(dictionary_s_tokens) != len(s_tokens):
        s = str.lower(s)
        #property_and_unit_tokens = [str.lower(token) for token in property_and_unit_tokens]
        s = preprocessing.solve_abbreviations(s, property_and_unit_tokens, input_is_string=True)  # need to adapt this and next method for just strings
        s = preprocessing.solve_similar_spelling(s, property_and_unit_tokens, input_is_string=True)
        s_tokens = set(s.split())

    user_property, user_unit = segment_user_string(s)
    return user_property, user_unit


#Runs training on the Naive Bayes classifier
def train_classifier():
    u = Classifier()
    u.train('my_unit.csv', t='u')
    p = Classifier()
    p.train('my_property.csv', t='p')


#Runs prediction on a saved Naive Bayes classifier model
def run_classifier_from_saved(s, t, predictions_to_return=10, ranked=False):
    c = Classifier()
    if ranked:
        return c.predict_top_x([s], predictions_to_return, t, load_model=True)
    else:
        return c.predict([s], t, load_model=True, have_return=True)


def app_run_classifier(s, t, model, ranked=False):
    if ranked:
        return model.predict_top_x([s], t)
    else:
        return model.predict([s], t)


#Tries to categorise user input as belonging to property or unit
def property_or_unit(s_tokens):
    unit_prob = len(s_tokens.intersection(unit_vocab))
    property_prob = len(s_tokens.intersection(property_vocab))
    #print("Unit probability:", unit_prob, " Property probability:", property_prob)
    if unit_prob > property_prob:
        return 'u'
    if property_prob > unit_prob:
        return 'p'
    else:
        return 1


def user_input_loop():
    while True:
        output = process_user_input()
        if output == 1:
            return
        else:
            properties, units = output
            if len(properties) > 0:
                property_predictions = run_classifier_from_saved(properties, 'p', ranked=True)
                print(property_predictions)
            if len(units) > 0:
                unit_predictions = run_classifier_from_saved(units, 'u', ranked=True)
                print(unit_predictions)


if __name__ == '__main__':
    # Process the qudt files first if haven't already
    #process_raw_qudt()

    # Run after making changes to training set datasets
    # Extract features from datasets without labels
    #resource_creation.extract_features_to_tag(datasets)

    # Tag untagged (taken from raw datasets) features by hand before running these
    # Extract features and labels from labelled datasets
    #resource_creation.tag_features('hand_tagged_unit.csv', 'unit')
    #resource_creation.tag_features('hand_tagged_property.csv', 'property')
    #resource_creation.tag_features('proc_qudt-property.csv', 'property')
    #resource_creation.tag_features('proc_qudt-unit.csv', 'unit')

    # Create the file for the binary classifier
    #resource_creation.create_binary_classification_file('my_property.csv', 'my_unit.csv')

    # Train classifiers
    #nn_train('property_or_unit.csv')
    #train_classifier()

    # Make predictions from user input
    user_input_loop()


