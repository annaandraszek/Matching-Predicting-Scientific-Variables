import unittest
import main

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
