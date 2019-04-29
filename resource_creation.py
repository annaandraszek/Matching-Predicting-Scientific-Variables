import pandas as pd
import preprocessing

# raw_file indicates whether to clean the file, save it, and create tokens, or only create tokens
def create_reference(file, raw_file=True): #creates a set of "dictionary" values from a qudt file (quantity or unit)
    path = 'raw datasets/'
    df = pd.read_csv(path+file, usecols=['rdfs:label'])

    if raw_file:
        df.replace('\\^\\^xsd:string', '', regex=True, inplace=True)
        df.replace("'", '', regex=True, inplace=True)
        if 'unit' in file:
            df = preprocessing.clean_table(df, has_measurements=False, has_units=True, units='rdfs:label')
        if 'property' in file:
            df = preprocessing.clean_table(df, measurements='rdfs:label')

        df.to_csv(path + 'proc_' + file, index=False)

    tokens = preprocessing.tokenise_column_values(df['rdfs:label'])
    tokens = tokens.union(set(['okta', 'hectopascal', 'micromole', 'from']))
    return tokens

