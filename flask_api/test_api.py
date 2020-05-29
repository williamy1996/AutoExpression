import numpy as np
import pandas as pd
import requests
train_url = 'http://0.0.0.0:3724/train_pipeline'
test_url = 'http://0.0.0.0:3724/serve_prediction'
models_url = 'http://0.0.0.0:3724//models'
train_files = {'data_file_X': open('./data/clyq_X.csv','rb'),'data_file_y': open('./data/clyq_y.csv','rb')}
r_train = requests.post(train_url, data={'model_name':'testclf1','objective':'clf'},files=train_files)
test_files = {'data_file_X': open('./data/clyq_X.csv','rb'),'data_file_y': open('./data/clyq_y.csv','rb')}
r_test = requests.post(test_url, data={'model_name':'testclf1','need_proba':True},files=train_files)
r_models = requests.post(models_url)
