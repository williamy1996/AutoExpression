import json
import time

import pandas as pd
import sklearn
import yaml
import werkzeug
from flask import Flask,render_template,url_for,request
from flask_restful import reqparse, Resource
import numpy as np
import pandas as pd
import argparse
import os
import sys
import time

from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split

sys.path.append(os.getcwd())
from flask_api.utilities import cross_validate, read_params
from solnml.utils.data_manager import DataManager
from solnml.estimators import Classifier
from werkzeug.datastructures import FileStorage

class Model(Resource):

    def __init__(self, model_factory):
        self.model_factory = model_factory
        self.parser = reqparse.RequestParser()
    
    def get(self):
        args = self.parser.parse_args()
        pipeline_id = request.form['model_name']
        pipeline = self.model_factory[pipeline_id]
        result = dict()
        
        result['model'] = pipeline['stats']
        
        mdl = pipeline['stats']['model']
        df = mdl.feature_origin()
        result['extract_features'] = df.to_json()
        return json.dumps(result)

class Models(Resource):

    def __init__(self, model_factory):
        self.model_factory = model_factory
    
    def get(self):
        model_ids = self.model_factory.pipelines.keys()
        result = dict()
        for m in model_ids:
            result[m] = self.model_factory[m]['stats']
        return json.dumps(result)



class Train(Resource):

    def __init__(self, model_factory):
        self.model_factory = model_factory
        self.parser = reqparse.RequestParser()
        #self.parser.add_argument('params', type=str,
            #location='files')
        self.parser.add_argument('data_file_X', type=FileStorage, 
            location='files')
        self.parser.add_argument('data_file_y', type=FileStorage,
            location='files')
    def post(self):
        start_time = time.time()
        args = self.parser.parse_args()
        X_file = request.files['data_file_X']
        y_file = request.files['data_file_y']
        _id = request.form['model_name']
        obj = request.form['objective']
        # read data
        X_train = np.array(pd.read_csv(X_file))
        y_train = np.array(pd.read_csv(y_file))[:,0]
        print(y_train)
        if not (obj):
            obj = 'clf'
        dm = DataManager(X_train, y_train)
        train_data = dm.get_data_node(X_train, y_train)
        
        save_dir = './data/eval_exps/soln-ml'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # train mode
        if(obj == 'clf'):
            mdl = Classifier(time_limit=100,
                    output_dir=save_dir,
                    ensemble_method='bagging',
                    evaluation='holdout',
                    metric='acc',
                    n_jobs=4)
        elif(obj == 'reg'):
            mdl = rgs = Regressor(metric='mse',
                    ensemble_method=ensemble_method,
                    evaluation=eval_type,
                    time_limit=time_limit,
                    output_dir=save_dir,
                    random_state=1,
                    n_jobs=n_jobs)
            
        mdl.fit(train_data)
        self.model_factory.add_pipeline(mdl, train_data, _id)
        print(self.model_factory)
        result = {'trainTime': time.time()-start_time, 
                  'trainShape': X_train.shape}
        self.model_factory[params['pipeline_id']]['stats'] = result
        return json.dumps(result)


class ServePrediction(Resource):

    def __init__(self, model_factory):
        self.model_factory = model_factory
        self.parser = reqparse.RequestParser()
        self.parser.add_argument('data_file_X', type=FileStorage,
                location='files')

    def post(self):
        args = self.parser.parse_args()
        X_file = request.files['data_file_X']
        X_test = np.array(pd.read_csv(X_file))
        y_test = np.zeros(X_test.shape[0])
        dm = DataManager(X_test, y_test)
        test_data = dm.get_data_node(X_test, y_test)
        _id = request.form['model_name']
        proba = request.form['need_proba']
        mdl = self.model_factory.pipelines[_id]['model']
        if(proba):
            try:
                y_pred = mdl.predict_proba(test_data)
            except:
                y_pred = mdl.predict(test_data)
        else:
            y_pred = mdl.predict(test_data)
        print(y_pred)
        #params = read_params(args['params'].stream)
        return pd.DataFrame(y_pred).to_json()

