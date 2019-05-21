import pandas as pd
import numpy as np
import preprocessing
import platform
import language_processing
import itertools

#Setting path to datasets - specific to user/pc
if 'FIJI-DP' in platform.uname()[1]:
    datasets_path = "C:/Users/AND522/Documents/Matching-Predicting-Scientific-Variables/raw datasets/"
else:
    datasets_path = "C:/Users/Anna/Documents/ml_approach/raw datasets/"


# Creates a set of tokens (words) contained in a file
# raw_file indicates whether to clean the file, save it, and create tokens, or only create tokens
def create_reference(file, raw_file=True):
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
        if 'unit' in file:
            df = preprocessing.clean_table(df, has_properties=False, units='rdfs:label', has_units=True, has_abbreviations=True)
        if 'property' in file:
            df = preprocessing.clean_table(df, properties='rdfs:label') #, units='qudt:unit', has_units=True)
        df.to_csv(path + 'proc_' + file, index=False)

    tokens = preprocessing.tokenise_column_values(df[col_to_tokenise])
    if 'qudt' in file:
        tokens = tokens.union({'okta', 'hectopascal', 'micromole', 'from'})
    #print(len(tokens), tokens)
    #print(len(tokens - language_processing.filter_stopwords(tokens)), tokens - language_processing.filter_stopwords(tokens))
    return language_processing.filter_stopwords(tokens)


def create_set_of_native(file):
    df = pd.read_csv(file)
    accepted_inputs = [line for line in df['native']] # {frozenset(line.split()) for line in df['native']}
    return accepted_inputs

def create_list_of_column(file, column):
    df = pd.read_csv(file)
    return [line for line in df[column]]

# Extracts unlabelled properties, units from raw datasets and saves to a file to label manually
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

    type = ['property', 'unit']
    dfs = [properties_df, units_df]
    for (t, df) in zip(type, dfs):
        try:
            tagged = pd.read_csv(path + 'hand_tagged_' + t + '.csv')
            df = tagged.append(df, sort=True)
            df.sort_values(by='class', inplace=True)
            df.drop_duplicates('native', inplace=True)
            df.to_csv(path+'hand_tagged_' + t + '.csv', index=False)

        except FileNotFoundError:
            print('Will make new hand_tagged_' + t + '.csv')
            units_df.to_csv(path+'hand_tagged_' + t + '.csv', index=False)



# Takes a dataframe which wants to be saved to my_property/my_unit and merges/saves it to it without duplication or erasure
def merge_with_my(df, type, sorting_col='class', native_col='native', prefix='my_'):
    try:
        my_df = pd.read_csv(prefix + type + '.csv')
        df = my_df.append(df, sort=True)

    except FileNotFoundError:
        print('Will make new ' + prefix + type + '.csv')
        df.drop_duplicates(native_col, inplace=True)
        df.to_csv(prefix + type + '.csv', index=False)
        return

    df.sort_values(by=sorting_col, inplace=True)
    df.drop_duplicates(native_col, inplace=True)
    df.dropna(subset=[native_col], inplace=True)
    df.to_csv(prefix + type + '.csv', index=False)



# Extracts properties, units from qudt files and saves to my_property/my_unit as themselves and their labels
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


def find_subsequences(sequence):
    subsequences = set()
    for i in range(1, len(sequence)):
        subsequences |= set(itertools.combinations(sequence, i))
    return list(subsequences)


def append_subsequences(df, col):
    subsequences = [] #pd.Series()
    for sample in df[col]:
        subsamples = find_subsequences(str(sample).split(' '))# pd.Series(find_subsequences(str(sample).split(' ')))
        subsamples = [sample for sample in subsamples if not all(x.isdigit() for x in sample)]
        subsequences.extend(subsamples)#, ignore_index=True)
    #df.native = df.native.append(subsequences, ignore_index=True)
    subsequences = list(set(subsequences))
    #subsequences = [list(subsequence) for subsequence in subsequences]
    subsequences = [" ".join(subsequence) for subsequence in subsequences]
    sdf = pd.DataFrame(subsequences, columns=['native']) #.transpose()
    df = pd.concat([df, sdf], ignore_index=True)
    return df


def create_binary_classification_file(property_file, unit_file):
    pdf = pd.read_csv(property_file, usecols=['native'])
    udf = pd.read_csv(unit_file, usecols=['native'])
    pdf = append_subsequences(pdf, 'native')
    udf = append_subsequences(udf, 'native')
    pdf['class'] = 'property'
    udf['class'] = 'unit'
    binary_df = pdf.append(udf)
    binary_df.dropna(inplace=True)
    binary_df.to_csv('property_or_unit.csv', index=False)


def create_display_files(proc_file, qudt_file, type):
    display_df = pd.DataFrame()# (columns=['processed_name', 'proper_name', 'url']) #, 'suggested_unit', 'suggested_unit_url'])
    proc_df = pd.read_csv(datasets_path + proc_file, usecols=['rdfs:label'])
    display_df['processed_name'] = proc_df.loc[:,'rdfs:label']

    if 'property' in type:
        qudt_df = pd.read_csv(datasets_path+qudt_file, usecols=['@id', 'rdfs:label', 'qudt:unit'])
    elif 'unit' in type:
        qudt_df = pd.read_csv(datasets_path+qudt_file, usecols=['@id', 'rdfs:label', 'qudt:abbreviation', 'qudt:symbol'])
    else:
        return 1

    display_df['url'] = qudt_df['@id']
    display_df['proper_name'] = qudt_df['rdfs:label']
    if 'property' in type:
        display_df['suggested_unit_url'] = qudt_df['qudt:unit']
    if 'unit' in type:
        display_df['abbreviation'] = qudt_df['qudt:symbol'].mask(qudt_df['qudt:symbol'].isna(), qudt_df['qudt:abbreviation'])

    # Clean qudt file cols of trash values
    if 'property' in type:
        display_df['suggested_unit'] = preprocessing.remove_web(qudt_df['qudt:unit'])
    display_df = preprocessing.remove_trash(display_df)

    if 'property' in type:
        display_df['suggested_unit'] = preprocessing.from_camelcase(display_df['suggested_unit'])
        display_df['suggested_unit'] = display_df['suggested_unit'].replace('_', ' ', regex=True)

    merge_with_my(display_df, 'display_' + type, sorting_col='processed_name', native_col='processed_name')


def add_to_user_training_set(user_property, user_unit, class_property, class_unit):
    user_str = user_property + ' (' + user_unit + ')'
    df = pd.DataFrame(columns=['native', 'property_class', 'unit_class'])
    df.native = np.asarray([user_str])
    df.property_class = np.asarray([class_property])
    df.unit_class = np.asarray([class_unit])
    merge_with_my(df, sorting_col='property_class', prefix='user_', type='trainingset')

def get_class_names():
    properties = create_list_of_column('my_display_property.csv', 'proper_name')
    units = create_list_of_column('my_display_unit.csv', 'proper_name')
    return properties, units