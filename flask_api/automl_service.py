# -*- coding: utf-8 -*-

import os
import json
import time
from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import argparse
import requests
import sklearn
import werkzeug
import resources
from solnml.utils.data_manager import DataManager
from solnml.estimators import Classifier
from solnml.estimators import Regressor
from werkzeug.datastructures import FileStorage

from flask_restful import reqparse, abort, Api, Resource
import yaml
import sys
from celery import Celery
sys.path.append('../')

app = Flask(__name__)
api = Api(app)
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'

celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)

model_factory = pd.read_csv('models_information/index.csv',index_col=0)

@celery.task
def model_fit(_id,mdl,train_data):
    info_path = './models_information/'+_id+'_information'
    info_file = open(info_path,'w')
    try:
        mdl.fit(train_data)
    except:
        info_file.write('Model training failed!')
        info_file.close()
        return -1
    result = dict()
    result['best_algo_id'] = str(mdl.best_algo_id)
    result['best_hpo_config'] = str(mdl.best_hpo_config)
    result['nbest_algo_id'] = str(mdl.nbest_algo_id)
    result['best_perf'] = str(mdl.best_perf)
    result['best_fe_config'] = str(mdl.best_fe_config)
    result['get_ens_model_info'] = str(mdl.get_ens_model_info)
    #get_ens_model_info is not realized in this version yet
    info_file.write(json.dumps(result))
    info_file.close()
    return 0


class Train(Resource):

    def __init__(self, model_factory):
        self.model_factory = model_factory
        self.parser = reqparse.RequestParser()
        self.parser.add_argument('data_file_X', type=FileStorage, 
            location='files')
        self.parser.add_argument('data_file_y', type=FileStorage,
            location='files')
        self.parser.add_argument('params_file', type=FileStorage,
            location='files')
    def post(self):
        start_time = time.time()
        args = self.parser.parse_args()
        _id = request.form['model_name']

        if(_id in list(self.model_factory.index)):
            return 'Error, model_name existed, please change model_name!'

        X_file = request.files['data_file_X']
        y_file = request.files['data_file_y']
        params_file = request.files['params_file']
        obj = request.form['objective']
        # read data
        X_train = np.array(pd.read_csv(X_file))
        y_train = np.array(pd.read_csv(y_file))[:,0]
        params = yaml.load(params_file)

        #print(y_train)
        if not (obj):
            obj = 'clf'
        dm = DataManager(X_train, y_train)
        train_data = dm.get_data_node(X_train, y_train)
        
        save_dir = '../data/eval_exps/soln-ml'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # train mode
        if(obj == 'clf'):
            mdl = Classifier(time_limit=params['time_limit'],
                    output_dir=save_dir,
                    ensemble_method=params['ensemble_method'],
                    evaluation=params['evaluation'],
                    metric=params['metric'],
                    n_jobs=4)

        elif(obj == 'reg'):
            mdl = rgs = Regressor(metric=params['metric'],
                    ensemble_method=params['ensemble_method'],
                    evaluation=params['evaluation'],
                    time_limit=params['time_limit'],
                    output_dir=save_dir,
                    random_state=1,
                    n_jobs=n_jobs)

        model_fit(_id,mdl,train_data)
        model_factory.index.append(pd.Index([_id]))
        model_factory.loc[_id,'info_path'] = './models_information/'+_id+'_information'
        model_factory.to_csv('models_information/index.csv')
        return ('Model '+_id+' is running!')

class Model_information(Resource):

    def __init__(self, model_factory):
        self.model_factory = model_factory
        self.parser = reqparse.RequestParser()
    
    def post(self):
        args = self.parser.parse_args()
        _id = request.form['model_name']
        if(_id not in list(self.model_factory.index)):
            return 'Error! No such model!'
        info_path = self.model_factory.loc[_id,'info_path']
        try:
            info_file = open(info_path,'r')
        except:
            return ('Model '+_id+' is running and not completed!')
        info = info_file.read()
        info_file.close()
        return info



api.add_resource(Train, '/train_pipeline',
    resource_class_kwargs={'model_factory': model_factory})
#api.add_resource(resources.ServePrediction, '/serve_prediction',
    #resource_class_kwargs={'model_factory': model_factory})
api.add_resource(Model_information, '/Model_information',
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
