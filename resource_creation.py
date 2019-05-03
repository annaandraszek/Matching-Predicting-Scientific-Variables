import pandas as pd
import numpy as np
import preprocessing
import platform

if 'FIJI-DP' in platform.uname()[1]:
    datasets_path = "C:/Users/AND522/Documents/Matching-Predicting-Scientific-Variables/raw datasets/"
else:
    datasets_path = "C:/Users/Anna/Documents/ml_approach/raw datasets/"

# raw_file indicates whether to clean the file, save it, and create tokens, or only create tokens
def create_reference(file, raw_file=True): #creates a set of "dictionary" values from a qudt file (quantity or unit)
    path = 'raw datasets/'
    if 'qudt-property' in file:
        df = pd.read_csv(path+file, usecols=['rdfs:label'])
        col_to_tokenise = 'rdfs:label'
    elif 'qudt-unit' in file:
        df = pd.read_csv(path+file, usecols=['rdfs:label', 'qudt:symbol', 'qudt:abbreviation'])
        col_to_tokenise = 'rdfs:label'

    else:
        df = pd.read_csv(file)
        col_to_tokenise = 'native'

    if raw_file:
        df.replace('\\^\\^xsd:string', '', regex=True, inplace=True)
        df.replace("'", '', regex=True, inplace=True)
        if 'unit' in file:
            df = preprocessing.clean_table(df, has_properties=False, units='rdfs:label', has_units=True, has_abbreviations=True)
        if 'property' in file:
            df = preprocessing.clean_table(df, properties='rdfs:label') #, units='qudt:unit', has_units=True)

        df.to_csv(path + 'proc_' + file, index=False)

    tokens = preprocessing.tokenise_column_values(df[col_to_tokenise])
    if 'qudt' in file:
        tokens = tokens.union({'okta', 'hectopascal', 'micromole', 'from'})
    return tokens


def extract_features_to_tag(datasets):
    path = 'raw datasets/'
    units = []
    properties = []

    for dataset in datasets:
        set_units, set_properties = preprocessing.raw_to_clean(datasets_path+dataset)
        if set_units:
            units = units + list(set_units)
        if set_properties:
            properties = properties + list(set_properties)

    properties = set(properties)
    units = set(units)

    units_df = pd.DataFrame(data=units, columns=['native'])
    properties_df = pd.DataFrame(data=properties, columns=['native'])
    try:
        tagged_units = pd.read_csv(path+'hand_tagged_unit.csv')
        units_df = tagged_units.append(units_df, sort=True)
        units_df.sort_values(by='class', inplace=True)
        units_df.drop_duplicates('native', inplace=True)
        units_df.to_csv(path+'hand_tagged_unit.csv', index=False)

    except FileNotFoundError:
        print('Will make new hand_tagged_unit.csv')
        units_df.to_csv(path+'hand_tagged_unit.csv', index=False)

    try:
        tagged_properties = pd.read_csv(path+'hand_tagged_property.csv')
        properties_df = tagged_properties.append(properties_df, sort=True)
        properties_df.sort_values(by='class', inplace=True)
        properties_df.drop_duplicates('native', inplace=True)
        properties_df.to_csv(path+'hand_tagged_property.csv', index=False)
    except FileNotFoundError:
        print('Will make new hand_tagged_property.csv')
        properties_df.to_csv(path+'hand_tagged_property.csv', index=False)

    #merge_with_my(units_df, 'unit')
    #merge_with_my(properties_df, 'property')


def merge_with_my(df, type):
    if type == 'unit':
        try:
            my_units = pd.read_csv('my_' + type + '.csv')
            df = my_units.append(df, sort=True)

        except FileNotFoundError:
            print('Will make new my_' + type + '.csv')
            df.drop_duplicates('native', inplace=True)
            df.to_csv('my_' + type + '.csv', index=False)
            return

    elif type == 'property':
        try:
            my_properties = pd.read_csv('my_' + type +'.csv')
            df = my_properties.append(df, sort=True)

        except FileNotFoundError:
            print('Will make new my_' + type + '.csv')
            df.drop_duplicates('native', inplace=True)
            df.to_csv('my_' + type + '.csv', index=False)
            return

    df.sort_values(by='class', inplace=True)
    df.drop_duplicates('native', inplace=True)
    df.dropna(subset=['native'], inplace=True)
    df.to_csv('my_' + type + '.csv', index=False)


def tag_features(dataset, t):
    old_df = pd.read_csv(datasets_path+dataset)

    if 'qudt' in dataset:
        df = pd.DataFrame()
        df = old_df[~old_df['rdfs:label'].str.contains('@en')]
        if 'unit' in t:
            new_df = pd.DataFrame()
            df['abbreviation'] = df['qudt:symbol'].combine_first(df['qudt:abbreviation'])
            new_df['native'] = np.concatenate((df['rdfs:label'], df['abbreviation']))
            new_df['class'] =  np.concatenate((df['rdfs:label'], df['rdfs:label']))
            merge_with_my(new_df, t)
        elif 'property' in t:
            df.rename({'rdfs:label': 'native'}, axis='columns', inplace=True)
            df['class'] = df['native']
            merge_with_my(df, t)
    else:
        #old_df = preprocessing.table_to_lower(old_df)
        merge_with_my(old_df, t)


def create_paired_reference(property_file, unit_file, pre_paired_file):
    property_df = pd.read_csv(property_file, usecols=['class']) #aka my property
    #unit_df = pd.read_csv(unit_file, usecols=['class']) # aka my unit
    pre_paired_df = pd.read_csv('raw datasets/' + pre_paired_file, usecols=['rdfs:label', 'qudt:unit']) # aka raw qudt-property

