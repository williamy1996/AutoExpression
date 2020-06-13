import numpy as np
import pandas as pd
import requests

## These need to be run on different bash
## redits-server
## celery worker -A flask_api/flask_celery.celery
## python flask_api/automl_service.py

train_url = 'http://0.0.0.0:3724/train_pipeline'
#test_url = 'http://0.0.0.0:3724/serve_prediction'
model_url = 'http://0.0.0.0:3724/Model_information'
train_files = {'data_file_X': open('./data/clyq_X.csv','rb'),'data_file_y': open('./data/clyq_y.csv','rb'),'params_file': open('./params/clf_params.yaml')}
r_train = requests.post(train_url, data={'model_name':'testclf2','objective':'clf'},files=train_files)
#test_files = {'data_file_X': open('./data/clyq_X.csv','rb'),'data_file_y': open('./data/clyq_y.csv','rb')}
#r_test = requests.post(test_url, data={'model_name':'testclf1','need_proba':True},files=train_files)
r_model = requests.post(model_url, data={'model_name':'testclf2'})
