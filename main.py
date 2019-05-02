import preprocessing
import resource_creation
#from ml_keras import NeuralNetwork
from machine_learning import Classifier
import pandas as pd
import numpy as np

datasets = ['AIMS_NingalooReef_AirTemperature_WindSpeed(Scalaravg10min)_WindDirection(VectorA-edited.csv',
               'ELI01m10m_201901.csv', 'IDCJDW4019.201903.csv', 'undownloadable.csv', ]
only_qudt_datasets = ['proc_qudt-property.csv', 'proc_qudt-unit.csv']

datasets_path = "C:/Users/Anna/Documents/ml_approach/raw datasets/"
#datasets_path = "C:/Users/AND522/Documents/ml_approach/raw datasets/"


def extract_features_to_tag(datasets):
    units = []
    measurements = []

    for dataset in datasets:
        set_units, set_measurements = preprocessing.raw_to_clean(datasets_path+dataset)
        if set_units:
            units = units + list(set_units)
        if set_measurements:
            measurements = measurements + list(set_measurements)

    measurements = set(measurements)
    units = set(units)

    units_df = pd.DataFrame(data=units, columns=['native'])
    measurements_df = pd.DataFrame(data=measurements, columns=['native'])
    merge_with_my(units_df, 'unit')
    merge_with_my(measurements_df, 'property')


def merge_with_my(df, type):
    if type == 'unit':
        try:
            my_units = pd.read_csv('my_' + type + '.csv')
            df = my_units.append(df, sort=True)

        except FileNotFoundError:
            print('Will make new my_units.csv')
            df.drop_duplicates('native', inplace=True)
            df.to_csv('my_' + type + '.csv', index=False)
            return

    elif type == 'property':
        try:
            my_measurements = pd.read_csv('my_' + type +'.csv')
            df = my_measurements.append(df, sort=True)

        except FileNotFoundError:
            print('Will make new my_measurements.csv')
            df.drop_duplicates('native', inplace=True)
            df.to_csv('my_' + type + '.csv', index=False)
            return

    df.sort_values(by='class', inplace=True)
    df.drop_duplicates('native', inplace=True)
    df.dropna(subset=['native'], inplace=True)
    df.to_csv('my_' + type + '.csv', index=False)


def tag_features(dataset, t):
    old_df = pd.read_csv(datasets_path+dataset)
    df = pd.DataFrame()

    df = old_df[~old_df['rdfs:label'].str.contains('@en')]
    if t == 'u':
        new_df = pd.DataFrame()
        df['abbreviation'] = df['qudt:symbol'].combine_first(df['qudt:abbreviation'])
        new_df['native'] = np.concatenate((df['rdfs:label'], df['abbreviation']))
        new_df['class'] =  np.concatenate((df['rdfs:label'], df['rdfs:label']))
        merge_with_my(new_df, 'unit')
    else:
        df.rename({'rdfs:label': 'native'}, axis='columns', inplace=True)
        df['class'] = df['native']
        merge_with_my(df, 'property')


# def nn_train_predict():
#     model = 'basic'
#     ml = NeuralNetwork('my_measurements.csv')
#     ml.train(model)
#     print(ml.predict(['wind'], model))
#     #print('humidity' in unit_terms, 'humidity' in measurement_terms)
#
#
# def nn_just_predict(string):
#     model = 'basic'
#     ml = NeuralNetwork('my_measurements.csv')
#     print(ml.predict([string], model))


def run_resource_creation():
    unit_terms = resource_creation.create_reference('qudt-unit.csv', raw_file=True)  # the set of all unique qudt unit words
    measurement_terms = resource_creation.create_reference('qudt-property.csv', raw_file=True)  # the set of all unique qudt property (measurement) words - less complete than unit set


def run_classifier():
    u = Classifier()
    u.train('my_unit.csv', t='u')

    m = Classifier()
    m.train('my_property.csv', t='m')

    print('Enter xxx to end')
    while True:
        predictions_to_return = 10
        s = str(input('Enter a string to predict:'))
        if s == 'xxx':
            break
        t = str(input('Enter p if property u if unit: '))
        if t == 'p':
            m.predict_top_x([s], predictions_to_return, 'm', load_model=True)
        elif t == 'u':
            u.predict_top_x([s], predictions_to_return, t, load_model=True)


if __name__ == '__main__':
    run_resource_creation()

    #Run after making changes to training sets
    extract_features_to_tag(datasets)
    tag_features('proc_qudt-property.csv', 'm')
    tag_features('proc_qudt-unit.csv', 'u')

    #Tag untagged (taken from raw datasets) features by hand before running classifier

    #Run to make predictions (using Complement Naive Bayes)
    #run_classifier()


