# -*- coding: utf-8 -*-
"""
automl_service.py
~~~~~~~~~~~~~~~~~

App implements an automl pipeline.

"""

import os
import json

from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import requests
import sklearn

import resources

from flask_restful import reqparse, abort, Api, Resource
from flask_api.utilities import read_params, ModelFactory


app = Flask(__name__)
api = Api(app)

model_factory = ModelFactory()


api.add_resource(resources.Train, '/train_pipeline',
    resource_class_kwargs={'model_factory': model_factory})
api.add_resource(resources.ServePrediction, '/serve_prediction',
    resource_class_kwargs={'model_factory': model_factory})
api.add_resource(resources.Models, '/models',
    resource_class_kwargs={'model_factory': model_factory})

# note: not used if using gunicorn
if __name__ == "__main__":
    if os.environ.get('VCAP_SERVICES') is None: # running locally
        PORT = 3724
        DEBUG = True
        print('DEBUG = True')
    else:                                       # running on CF
        PORT = int(os.getenv("PORT"))
        DEBUG = False
        print('DEBUG = False')

    app.run(host='0.0.0.0', port=PORT, debug=DEBUG)
