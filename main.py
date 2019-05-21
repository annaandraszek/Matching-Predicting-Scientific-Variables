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
    global binary_model
    model = 'binary'
    binary_model = NeuralNetwork(file)
    binary_model.train(model)


def nn_train_predict(file):
    global binary_model
    #model = 'binary'
    binary_model = NeuralNetwork(file)
    binary_model.train()
    test_xs = ['wind', 'degree', 'horsepower', 'hp water', 'foot', 'concentration']
    print(binary_model.predict(test_xs))


def app_binary_predict(string):
    global binary_model
    return binary_model.predict(string)


def app_class_predict(model, string, t):
    return model.predict_top_x([string], t)


def app_load_models():
    global binary_model
    binary_model = NeuralNetwork('property_or_unit.csv')
    binary_model.load_model_from_file()
    global property_model
    property_model = Classifier()
    property_model.load_model_and_classes('p')
    global unit_model
    unit_model = Classifier()
    unit_model.load_model_and_classes('u')
    #return binary_model, properties_model, units_model


#Cleans up/pre-processes the raw qudt datasets and saves as files in only_qudt_datasets
def process_raw_qudt():
    for dataset in raw_qudt_datasets:
        resource_creation.create_reference(dataset, raw_file=True)


def segment_user_string(string):
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
    labels, probabilities = app_binary_predict(segments)
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


def app_process_user_input(s):
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

    user_property, user_unit = segment_user_string(s)
    return user_property, user_unit


def app_user_input(s):
    global property_model
    global unit_model
    output = app_process_user_input(s)
    if output == 1:
        return
    else:
        properties, units = output
        if len(properties) > 0 and len(units) <= 0:
            property_predictions = app_run_classifier(properties, 'p', property_model, ranked=True)
            return ('p', find_more_informative_result(property_predictions, 'property')), ('None',[])
        if len(units) > 0 and len(properties) <=0:
            unit_predictions = app_run_classifier(units, 'u', unit_model, ranked=True)
            return ('u', find_more_informative_result(unit_predictions, 'unit')), ('None', [])
        elif len(units) > 0 and len(properties) > 0:
            property_predictions = app_run_classifier(properties, 'p', property_model, ranked=True)
            unit_predictions = app_run_classifier(units, 'u', unit_model, ranked=True)
            return ('p', find_more_informative_result(property_predictions, 'property')), ('u', find_more_informative_result(unit_predictions, 'unit'))
        else: return "No vocabulary terms found"


def find_more_informative_result(results, t):
    ref_df = pd.read_csv('my_display_'+ t +'.csv')
    informative_results = pd.DataFrame()
    for result in results:
        print(result)
        informative_results = informative_results.append(ref_df.loc[ref_df['processed_name'] == result[1]], ignore_index=True)
    return informative_results

#Runs training on the Naive Bayes classifier
def train_classifier():
    global unit_model
    unit_model = Classifier()
    unit_model.train('my_unit.csv', t='u')
    global property_model
    property_model = Classifier()
    property_model.train('my_property.csv', t='p')


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
        s = str(input('Enter a string to predict:'))
        if s == 'xxx':
            return
        predictions = app_user_input(s)
        print(predictions)

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

    #resource_creation.create_display_files('proc_qudt-property.csv', 'qudt-property.csv', 'property')
    #resource_creation.create_display_files('proc_qudt-unit.csv', 'qudt-unit.csv', 'unit')

    # Train classifiers
    #nn_train('property_or_unit.csv')
    train_classifier()

    # Make predictions from user input
    app_load_models()
    user_input_loop()
