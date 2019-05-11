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


def nn_train_predict(file):
    model = 'binary'
    ml = NeuralNetwork(file)
    ml.train(model)
    test_xs = ['wind', 'degree', 'horsepower', 'hp water', 'foot', 'concentration']
    for x in test_xs:
        print(ml.predict([x], model))


def nn_just_predict(string):
    model = 'binary'
    ml = NeuralNetwork('property_or_unit.csv')
    print(ml.predict([string], model))


#Cleans up/pre-processes the raw qudt datasets and saves as files in only_qudt_datasets
def process_raw_qudt():
    for dataset in raw_qudt_datasets:
        resource_creation.create_reference(dataset, raw_file=True)


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


    # attempt to replace the below with a method for segmenting the types of input
    #accepted_inputs_p = resource_creation.create_set_of_native('my_property.csv')
    #accepted_inputs_u = resource_creation.create_set_of_native('my_unit.csv')

    # inputs = s.split()
    # pair_scores = []
    # for i in range(len(inputs)-1):  # search for side-by-side tokens in accepted sets to
    #     w1, w2 = inputs[i], inputs[i+1]
    #     p_score, u_score = 0, 0
    #     for set in accepted_inputs_p:
    #         if w1 in set and w2 in set:
    #             p_score += 1
    #             break   # only increment once per pair of words to avoid sample bias
    #     for set in accepted_inputs_u:
    #         if w1 in set and w2 in set:
    #             u_score +=1
    #             break
    #
    #     pair_scores.append(())

    t = property_or_unit(s_tokens)
    if t == 1:
        t = str(input('Enter p if property u if unit: '))
    if t == 'p' or t == 'u':
        return s, t
    else:
        print("Please enter 'p' for property or 'u' for unit or 'xxx' to exit")
        return 0



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
        c.predict_top_x([s], predictions_to_return, t, load_model=True)
    else:
        return c.predict([s], t, load_model=True, have_return=True)


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
        t = process_user_input()
        if t == 1:
            break
        if t == 0:
            continue
        else:
            user_string, type = t
            #run_classifier() # would recommend against running this method as-is right now - as it would be
                                # re-trained on every user input
                                # run if want to re-train the model before making predictions
            #print(run_classifier_from_saved(user_string, type)) # run if want to use a pre-trained model to make predictions
            run_classifier_from_saved(user_string, type, ranked=True)


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

    # User input-prediction loop
    # Run to make predictions (using Complement Naive Bayes)

    #resource_creation.create_binary_classification_file('my_property.csv', 'my_unit.csv')
    nn_train_predict('property_or_unit.csv')
    #test_xs = ['wind', 'degree', 'horsepower', 'hp water', 'foot', 'concentration']
    #for x in test_xs:
    #    nn_just_predict(x)
    #while True:
    #    string = str(input('Enter a string to categorise: '))
    #    nn_just_predict(string)
    #train_classifier()
    #user_input_loop()


