#!/usr/bin/env python
from flask import Flask, request, redirect, url_for, render_template
import pandas as pd
import requests
import numpy as np

backend_uri = 'http://mv-server:5001'

app = Flask(__name__)

@app.route('/predict/<var>', methods=['GET'])
def predict(var):
    data = {'input': var}
    response = requests.post(url=backend_uri+"/predict", json=data).json()
    print(response)
    result = response['result']

    print(result, len(result))
    if isinstance(result, str):
        return render_template('index.html', no_prediction=result)

    property_pred = pd.read_json(result['p'])
    property_pred.fillna(value=np.nan, inplace=True)

    unit_pred = pd.read_json(result['u'])
    property_pred.fillna(value=np.nan, inplace=True)

    print_sugg_units = False
    if unit_pred.empty:
        print_sugg_units = True
    return render_template('index.html', prediction=(('p', property_pred), ('u', unit_pred)), suggested_units=print_sugg_units, zip=zip, var=var)


@app.route('/')
@app.route('/predict', methods=['POST','GET'])
def empty_predict():
    if request.method == 'POST':
        var = request.form['var']
        return redirect(url_for('predict', var=var))
    else:
        return render_template('index.html', prediction=False)


@app.route('/submit', methods=['POST', 'GET'])
def submit():
    if request.method == 'POST':
        result = requests.get(url=backend_uri+"/classnames").json()
        properties = result['properties']
        units = result['units']

        class_unit = None
        class_property = None
        user_property = request.form['user_property']
        if user_property:
            try:
                class_property = request.form['class_property']
            except KeyError:
                return render_template('submit.html', properties=properties, units=units, message="Please select a class for your property.")
        else: user_property = None

        user_unit = request.form['user_unit']
        if user_unit:
            try:
                class_unit = request.form['class_unit']
            except KeyError:
                return render_template('submit.html', properties=properties, units=units, message="Please select a class for your unit.")
        else:
            user_unit = None

        data = {'user_property': user_property, 'user_unit':user_unit, 'class_property':class_property, 'class_unit':class_unit}
        if not (user_property or user_unit):
            return render_template('submit.html', properties=properties, units=units, message="Error: Can't submit no data.")
        requests.post(url=backend_uri+"/submit", data=data)

        success_message = "Submission of {} successful.".format(data)
        return render_template('submit.html', properties=properties, units=units, message=success_message)
    else:
        result = requests.get(url=backend_uri+"/classnames").json()
        properties = result['properties']
        units = result['units']
        return render_template('submit.html', properties=properties, units=units)

# @app.route('/')
# def home():
#     return 'Hi! Go to /predict to enter a scientific variable'
#

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)