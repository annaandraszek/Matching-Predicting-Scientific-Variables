import pandas as pd
import numpy as np
import stringdist
import re
import math

import resource_creation

# pt 1: take completely raw datasets and create several tables
    # drop unecessary cols and do pre-processing on text

def raw_to_clean(filename): #this method will export pre-processed sets of native property and unit terms, which
    unit_col = 'units'
    property_col = 'parameter'
    has_units = True
    has_properties = True
    # can be merged with sets of these from other files to create a table which will be manually tagged
    # not a method you'd use each time, as it requires manual work after
    if 'NingalooReef' in filename:
        old_df = pd.read_csv(filename, usecols=[property_col, unit_col])
        df = pd.DataFrame()
        for column in old_df:  # filter out duplicates here for speed
            #print(old_df[column].unique())
            df[column] = pd.Series(old_df[column].unique())

    elif 'IDCJDW' in filename:
        old_df = pd.read_csv(filename, skiprows=8, encoding='unicode_escape')
        old_df.drop(['Date'], axis=1, inplace=True)

        properties = []
        for col in old_df:
            properties.append(col)
        units, properties = segment_properties_units(properties, 'p')
        df = pd.DataFrame(data={unit_col: units, property_col: properties})

    elif 'ELI01m10m' in filename:
        old_df = pd.read_csv(filename)
        old_df.drop(['Date Time', 'Unnamed: 10', 'Unnamed: 11'], axis=1, inplace=True)

        properties = []
        for col in old_df:
            properties.append(col)
        units, properties = segment_properties_units(properties, 'lw')
        df = pd.DataFrame(data={unit_col: units, property_col: properties})

    elif 'undownloadable' in filename:
        old_df = pd.read_csv(filename, usecols=['string'])
        units, properties = segment_properties_units(old_df['string'], 'p')
        df = pd.DataFrame(data={'units': units, property_col: properties})

    elif 'qudt' in filename:  # this section assumes files containing property and units are mutually exclusive. change if otherwise
        if 'proc' not in filename:
            print ('Input name of processed file. Run resource_creation.create_reference() on file first.')
            return 1
        df = pd.read_csv(filename)
        if 'property' in filename:
            unit_col = None
            has_units = False
            property_col = 'rdfs:label'

        elif 'unit' in filename:
            has_property = False
            property_col = None
            unit_col = 'rdfs:label'

    else:
        print('Enter a valid filename')
        return 1
    df = table_to_lower(df)

    df = clean_table(df, property_col, has_property, unit_col, has_units)

    #create a set of native property/unit names
    if unit_col:
        native_units = pd.Series.unique(df[unit_col].dropna())
        unit_terms = resource_creation.create_reference('proc_qudt-unit.csv', raw_file=False)
        native_units = solve_abbreviations(native_units, unit_terms)
        native_units = solve_similar_spelling(native_units, unit_terms)
        native_units = [x.strip() for x in native_units]

    if property_col:
        native_properties = pd.Series.unique(df[property_col])
        native_properties = [x.strip() for x in native_properties] # remove whitespace which can cause them to not be identified as duplicates

    if unit_col and property_col:
        return native_units, native_properties
    elif unit_col and not property_col:
        return native_units, None
    elif property_col and not unit_col:
        return None, native_properties

def tokenise_column_values(column):
    return set(column.str.split(' ', expand=True).stack().unique())


def segment_properties_units(raw_strings, segment_on='p'): #assume units in parentheses
    units = []
    properties = []
    if segment_on == 'p': #parentheses
        for raw in raw_strings:
            if '(' in raw:
                p, u = raw.rsplit('(', 1)
                u = u.rstrip(')')
            else:
                p = raw
                u = np.nan
            units.append(u)
            properties.append(p)
    elif segment_on == 'lw': # last word
        for raw in raw_strings:
            p, u = raw.rsplit(' ', 1)
            units.append(u)
            properties.append(p)

    return units, properties

def solve_similar_spelling(units, unit_terms, max_distance=2):
    for s in range(len(units)):
        p = units[s].split(' ')
        for w in p:
            if not w in unit_terms:
                best_u = ''
                best_udist = max_distance
                for u in unit_terms:
                    dist = stringdist.rdlevenshtein(w, u)
                    if dist < len(w) - 1:
                        if dist <= best_udist:
                            best_udist = dist
                            best_u = u
                            #print(w, u, dist)
                if best_u != '':
                    units[s] = units[s].replace(w, best_u)
    return units


def solve_abbreviations(units, unit_terms): #takes the unit tokens #todo: do I also want to transform the unit token set?
    unit_abbrev_dict = {'deg':'degree', 'c':'celsius', '¡c':'degree celsius', 'hpa':'hectopascal', 'km':'kilometer', 'm':'meter',
                        'm2':'square meter', 's':'second', 'μmol':'micromole', 'mcmol':'micromole', 'h':'hour', 'mm':'millimeter',
                        'w':'watt', 'dir':'direction', 'mj': 'megajoule'}

    for s in range(len(units)):
        p = units[s].split(' ')
        for w in p:
            if not w in unit_terms:
                try:
                    value = unit_abbrev_dict[w]
                    #print('before ', units[s])
                    units[s] = units[s].replace(w, value)
                    #print('after ', units[s])

                except KeyError:
                    print('no abbreviation for', w)
                    #StopIteration  # if the word/abbreviation doesn't exist in the unit or abbreviation dictionary
    return units


def clean_table(table, properties='parameter', has_properties=True, units='units', has_units=False, has_abbreviations=False):
    table.replace('http://registry.it.csiro.au/def/environment/unit/', '', regex=True, inplace=True)
    table.replace('http://qudt.org/vocab/unit#', '', regex=True, inplace=True)
    table.replace('http://', '', regex=True, inplace=True)
    table.replace('<', '', regex=True, inplace=True)
    table.replace('>', '', regex=True, inplace=True)
    table.replace('\\|', ' or ', regex=True, inplace=True)

    table.replace('_', ' ', regex=True, inplace=True)
    table.replace('\\(', '', regex=True, inplace=True)
    table.replace('\\)', '', regex=True, inplace=True)
    table.replace('\\[', '', regex=True, inplace=True)
    table.replace('\\]', '', regex=True, inplace=True)
    table.replace('/', ' / ', regex=True, inplace=True)
    table.replace('-', ' ', regex=True, inplace=True)
    table.replace(',', ' ', regex=True, inplace=True)
    table.replace('   ', ' ', regex=True, inplace=True)
    table.replace('\\^ ', '^', regex=True, inplace=True)
    table.replace('\\^', ' ^', regex=True, inplace=True)
    table.replace('\\^1', '', regex=True, inplace=True)
    table.replace('\\*', ' * ', regex=True, inplace=True)

    if has_units:
        if has_abbreviations:
            table[units] = from_camelcase(table[units])
        table[units].replace('/', 'per', regex=True, inplace=True) # if the unit column isn't called 'units' either change earlier or pass in as arg
        table[units].replace('%', 'percent', regex=True, inplace=True)
        table[units].replace('°', 'degree ', regex=True, inplace=True)
    #if has_abbreviations: #would need fixing, probably undesired
        #new = [from_camelcase([x]) if isinstance(x, str) and ('deg' in x or 'Deg' in x) else x for x in table['qudt:symbol']]
        #table['qudt:symbol'] = table['qudt:symbol'].where('deg' not in table['qudt:symbol'], from_camelcase(table['qudt:symbol']))
        #table['qudt:abbreviation'] = table['qudt:abbreviation'].where('deg' not in table['qudt:abbreviation'] or 'Deg' not in table['qudt:abbreviation'], from_camelcase(table['qudt:abbreviation']))

    table.replace('DegC', 'degC', inplace=True)
    table = table_to_lower(table)
    return table


def from_camelcase(array):
    us = []
    for unit in array:
        if unit == 'pH':
            us.append('pH')
            continue
        elif isinstance(unit, float) and math.isnan(unit):
            us.append(np.nan)
            continue
        elif unit != 'pH' and not isinstance(unit, float):
            #unit = unit.replace('_', ' ')
            unit = re.sub('(?!^)([A-Z][a-z]+)', r' \1', str(unit)).split()
            us.append(' '.join(unit))
    return us


def table_to_lower(table):
    for column in table:
        if 'symbol' not in column and 'abbreviation' not in column:
            table[column] = table[column].str.lower()
    return table
