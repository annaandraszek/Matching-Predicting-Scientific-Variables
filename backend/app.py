#!/usr/bin/env python
from flask import Flask, request
from flask_restful import Resource, Api
from main import app_load_models, app_user_input
from resource_creation import add_to_user_training_set, get_class_names
import requests

app = Flask(__name__)
api = Api(app)

# needed functions for web app: main.app_user_input
# resource_creation.add_to_user_training_set, get_class_names

class Predict(Resource):
    def post(self):
        data = request.get_json(force=True)
        input = data['input']
        result = app_user_input(str(input))
        if not isinstance(result, str):  # result will be returned as a tuple
            properties, units = result
            p_key, p_df = properties
            u_key, u_df = units
            p_json = p_df.to_json()
            u_json = u_df.to_json()
            result = {'p': p_json, 'u': u_json}
            return {'result': result}
        else:
            return {'result': result}


class AddToUserTrainingSet(Resource):
    def post(self):
        data = request.get_json(force=True)
        user_property = data['user_property']
        user_unit = data['user_unit']
        class_property = data['class_property']
        class_unit = data['class_unit']
        add_to_user_training_set(user_property, user_unit, class_property, class_unit)
        return {'result': 'Addition to training set successful'}


class GetClassNames(Resource):
    def get(self):
        properties, units = get_class_names()
        return {'properties': properties, 'units': units}

        
api.add_resource(Predict, '/predict', endpoint='predict')
api.add_resource(AddToUserTrainingSet, '/submit', endpoint='submit')
api.add_resource(GetClassNames, '/classnames', endpoint='classnames')


if __name__ == '__main__':
    app_load_models()
    app.run(host='0.0.0.0', port=5001, debug=True)
    