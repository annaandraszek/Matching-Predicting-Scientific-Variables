import pandas as pd
import preprocessing

# raw_file indicates whether to clean the file, save it, and create tokens, or only create tokens
def create_reference(file, raw_file=True): #creates a set of "dictionary" values from a qudt file (quantity or unit)
    path = 'raw datasets/'
    if 'property' in file:
        df = pd.read_csv(path+file, usecols=['rdfs:label'])
    elif 'unit' in file:
        df = pd.read_csv(path+file, usecols=['rdfs:label', 'qudt:symbol', 'qudt:abbreviation'])

    if raw_file:
        df.replace('\\^\\^xsd:string', '', regex=True, inplace=True)
        df.replace("'", '', regex=True, inplace=True)
        if 'unit' in file:
            df = preprocessing.clean_table(df, has_measurements=False, units='rdfs:label', has_units=True) #has_abbreviations=True
        if 'property' in file:
            df = preprocessing.clean_table(df, measurements='rdfs:label') #, units='qudt:unit', has_units=True)

        df.to_csv(path + 'proc_' + file, index=False)

    tokens = preprocessing.tokenise_column_values(df['rdfs:label'])
    tokens = tokens.union({'okta', 'hectopascal', 'micromole', 'from'})
    return tokens

def create_paired_reference(property_file, unit_file, pre_paired_file):
    property_df = pd.read_csv(property_file, usecols=['class']) #aka my property
    #unit_df = pd.read_csv(unit_file, usecols=['class']) # aka my unit
    pre_paired_df = pd.read_csv('raw datasets/' + pre_paired_file, usecols=['rdfs:label', 'qudt:unit']) # aka raw qudt-property

