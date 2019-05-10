import unittest
import main
import resource_creation
import machine_learning
import language_processing

class TestClassifier(unittest.TestCase):

    def test_units_prediction(self):
        inputs = [('hr','hour'),
                  ('cm','centimeter'),
                  ('hp h2o', 'horsepower water')]
        for input in inputs:
            #print(input, input[0], input[1], (main.run_classifier_from_saved(input[0], 'u')))
            self.assertEqual((main.run_classifier_from_saved(input[0], 'u'))[0], input[1])

    def test_properties_prediction(self):
        inputs = [('temperature 10cm', 'subsurface temperature')]
        for input in inputs:
            self.assertEqual((main.run_classifier_from_saved(input[0], 'p'))[0], input[1])

    def test_parameter_tuning(self):
        c = machine_learning.Classifier()
        c.train('my_unit.csv', 'u')
        c.parameter_tuning()
        c.train('my_property.csv', 'p')
        c.parameter_tuning()


class TestResourceCreation(unittest.TestCase):

    def test_accepted_inputs(self):
        accepted_inputs = resource_creation.create_set_of_native('my_property.csv')
        print(accepted_inputs)


class TestProcessing(unittest.TestCase):

    def test_tokenise(self):
        inputs = ['C10 - C14 hydrocarbon fraction concentration', 'benzo[b]fluoranthene concentration',
                  '2-chlorophenol-3,4,5,6-D4 concentration']
        for input in inputs:
            print(language_processing.tokenise(input))

    def test_punkt_tokenise(self):
        inputs = ['C10 - C14 hydrocarbon fraction concentration', 'benzo[b]fluoranthene concentration',
                  '2-chlorophenol-3,4,5,6-D4 concentration']
        for input in inputs:
            print(language_processing.punkt_tokenise(input))

    def test_filter_stopwords(self):
        inputs = [['sulphide', 'as', 'S', 'concentration'],
                  ['Pound', 'per', 'Cubic', 'Yard'],
                  ['2', '-', 'chlorophenol', '-', '3', ',', '4', ',', '5', ',', '6', '-', 'D4', 'concentration']]
        for input in inputs:
            print(language_processing.filter_stopwords(input))

    def test_find_synonymns(self):
        inputs = ['concentration', 'pound', 'meter', 'metre', 'square', 'cubic', 'wind', 'hr', 'per']
        for input in inputs:
            try:
                print(language_processing.find_synonymns(input))
            except IndexError:
                continue

    def test_find_stems(self):
        inputs = ['concentration', 'pound', 'metre', 'meters', 'square', 'cubic', 'wind', 'hour', 'hr', 'per', 'percent']
        print([language_processing.find_stem(input) for input in inputs])

    def test_find_lemmas(self):  # will return proper words, unlike the above
        inputs = ['concentration', 'pound', 'metre', 'meters', 'square', 'cubic', 'wind', 'hour', 'hr', 'per', 'percent']
        print([language_processing.find_lemma(input) for input in inputs])


    def test_fix_spelling(self):
        inputs = ['wnd', 'wind', 'fahrenheit', 'farenheit', 'metre', 'meters']
        print([language_processing.fix_spelling(input) for input in inputs])

