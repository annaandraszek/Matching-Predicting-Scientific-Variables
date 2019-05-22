from flask import Flask, request, redirect, url_for, render_template
app = Flask(__name__)
from main import app_user_input, app_load_models
from resource_creation import add_to_user_training_set, get_class_names
app_load_models()
import pandas as pd

@app.route('/result/<variable>')
def result(variable):
    return 'Predicting %s' % variable

@app.route('/')
@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        inpt = request.form['var']
        result = app_user_input(str(inpt))
        print(result, len(result))
        if isinstance(result, str):
            return render_template('index.html', no_prediction=result)
        #if len(result) > 2:
        #    result = [result]

        property_pred = pd.DataFrame()
        unit_pred = pd.DataFrame()

        for predictions in result:
            if predictions[0] == 'p':
                property_pred = predictions[1]
            elif predictions[0] == 'u':
                unit_pred = predictions[1]
            else:
                continue

        print_sugg_units = False
        if unit_pred.empty:
            print_sugg_units = True

        return render_template('index.html', prediction=(('p', property_pred), ('u', unit_pred)), suggested_units=print_sugg_units, zip=zip)
    else:
        return render_template('index.html', prediction=False)

@app.route('/submit', methods=['POST', 'GET'])
def submit():
    if request.method == 'POST':
        properties, units = get_class_names()

        class_unit = None
        class_property = None
        user_property = request.form['user_property']
        if user_property:
            try:
                class_property = request.form['class_property']
            except KeyError:
                return render_template('submit.html', properties=properties, units=units, message="Please select a class for your property.")

        user_unit = request.form['user_unit']
        if user_unit:
            try:
                class_unit = request.form['class_unit']
            except KeyError:
                return render_template('submit.html', properties=properties, units=units, message="Please select a class for your unit.")

        if class_unit and class_property:
            add_to_user_training_set(user_property, user_unit, class_property, class_unit)
        elif class_unit and not class_property:
            add_to_user_training_set('', user_unit, '', class_unit)
        elif class_property and not class_unit:
            add_to_user_training_set(user_property, '', class_property, '')

        return render_template('submit.html', properties=properties, units=units, message="Submission successful.")
    else:
        properties, units = get_class_names()
        return render_template('submit.html', properties=properties, units=units)



#def home():
#    return 'Hi! Go to /predict to enter a scientific variable'


if __name__ == '__main__':
    app.run(debug=True)
