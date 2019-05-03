import preprocessing
import resource_creation
#from ml_keras import NeuralNetwork
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


# def nn_train_predict():
#     model = 'basic'
#     ml = NeuralNetwork('my_property.csv')
#     ml.train(model)
#     print(ml.predict(['wind'], model))
#
#
# def nn_just_predict(string):
#     model = 'basic'
#     ml = NeuralNetwork('my_property.csv')
#     print(ml.predict([string], model))

#Cleans up/pre-processes the raw qudt datasets and saves as files in only_qudt_datasets
def process_raw_qudt():
    for dataset in raw_qudt_datasets:
        resource_creation.create_reference(dataset, raw_file=True)


#Runs training and prediction on the Naive Bayes classifier
def run_classifier():
    u = Classifier()
    u.train('my_unit.csv', t='u')
    p = Classifier()
    p.train('my_property.csv', t='p')

    # Input loop for user to provide strings for prediction
    print('Enter xxx to end')
    while True:
        predictions_to_return = 10
        s = str(input('Enter a string to predict:'))
        if s == 'xxx':
            break
        t = property_or_unit(s)
        if t == 1:
            t = str(input('Enter p if property u if unit: '))
        if t == 'p':
            p.predict_top_x([s], predictions_to_return, t)
        elif t == 'u':
            u.predict_top_x([s], predictions_to_return, t)
        else:
            print("Please enter 'p' for property or 'u' for unit or 'xxx' to exit")


#Runs prediction on a saved Naive Bayes classifier model
def run_classifier_from_saved():
    c = Classifier()
    print('Enter xxx to exit')

    # Input loop for user to provide strings for prediction
    while True:
        predictions_to_return = 10
        s = str(input('Enter a string to predict:'))
        if s == 'xxx':
            break
        t = property_or_unit(s)
        if t == 1:
            t = str(input('Enter p if property u if unit: '))
        if t == 'p' or t == 'u':
            c.predict_top_x([s], predictions_to_return, t, load_model=True)
        else:
            print("Please try again and enter 'p' for property or 'u' for unit or 'xxx' to exit")


#Tries to categorise user input as belonging to property or unit
def property_or_unit(input):
    s_tokens = set(input.split())
    unit_vocab = resource_creation.create_reference('my_unit.csv', raw_file=False)
    property_vocab = resource_creation.create_reference('my_property.csv', raw_file=False)
    unit_prob = len(s_tokens.intersection(unit_vocab))
    property_prob = len(s_tokens.intersection(property_vocab))
    #print("Unit probability:", unit_prob, " Property probability:", property_prob)
    if unit_prob > property_prob:
        return 'u'
    if property_prob > unit_prob:
        return 'p'
    else:
        return 1


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

    # Make predictions (using Complement Naive Bayes) - choose one of the two methods below
    run_classifier()  # run if want to re-train the model before making predictions
    #run_classifier_from_saved()  # run if want to use a pre-trained model to make predictions



