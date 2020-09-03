import argparse
import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
import json
import pickle as pkl
from sklearn.datasets import load_iris
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from solnml.utils.data_manager import DataManager
from solnml.estimators import Classifier
from solnml.utils import saveloadmodel

class Ensemble_models:
    def __init__(self,ensemble_info,mdl_list):
        self.ensemble_info = ensemble_info
        self.model_list = mdl_list
    def predict_proba(self,test_x):
        print('This is \'predict_proba\'. For regression model, please run \'predict\'.')
        
        if(self.ensemble_info['ensemble_method']=='none'):
            mdl0 = self.model_list[0].replace('\n','')
            base_model = pickle.load(open(mdl0,'rb'))
            y_predict = base_model.predict_proba(test_x)
            return y_predict
        
        if(self.ensemble_info['ensemble_method']=='bagging'):
            y_predict = []
            for mdl in self.model_list:
                mdl = mdl.replace('\n','')
                base_model = pickle.load(open(mdl,'rb'))
                y_predict.append(base_model.predict_proba(test_x))
            y_predict = np.array(y_predict)
            return np.average(y_predict,axis=0)
        
        if(self.ensemble_info['ensemble_method']=='ensemble_selection'):
            y_predict = []
            weights = np.array(pd.read_json(self.ensemble_info['ensemble_weights']))[:,0]
            i = 0
            for mdl in self.model_list:
                mdl = mdl.replace('\n','')
                base_model = pickle.load(open(mdl,'rb'))
                y_predict.append(base_model.predict_proba(test_x)*weights[i])
                i+=1
            y_predict = np.array(y_predict)
            return np.sum(y_predict,axis=0)
        
        if(self.ensemble_info['ensemble_method']=='stacking'):
            meta_learner = pickle.load(open(self.ensemble_info['meta_learner_path'],'rb'))
            kfold = self.ensemble_info['kfold']
            mdl0 = self.model_list[0].replace('\n','')
            base_model = pickle.load(open(mdl0,'rb'))
            y_predict = base_model.predict_proba(test_x)
            n_dim = y_predict.shape[1]
            sample_dim = y_predict.shape[0]
            y_predict = []
            if(n_dim==2):
                n_dim = 1
            i=0
            for mdl in self.model_list:
                if(i == 0):
                    new_sumpredict = np.zeros([sample_dim,n_dim])
                mdl = mdl.replace('\n','')
                base_model = pickle.load(open(mdl,'rb'))
                new_predict = base_model.predict_proba(test_x)
                if(n_dim==1):
                    new_predict = new_predict[:,1:]
                new_sumpredict = new_sumpredict + new_predict/kfold
                i+=1
                if(i==kfold):
                    i=0
                    y_predict.append(new_sumpredict)
            
            y_predict = np.hstack(y_predict)
            y_pred = meta_learner.predict_proba(y_predict)
            return y_pred
        
        if(self.ensemble_info['ensemble_method']=='blending'):
            meta_learner = pickle.load(open(self.ensemble_info['meta_learner_path'],'rb'))
            mdl0 = self.model_list[0].replace('\n','')
            base_model = pickle.load(open(mdl0,'rb'))
            y_predict = base_model.predict_proba(test_x)
            n_dim = y_predict.shape[1]
            if(n_dim==2):
                n_dim = 1
            y_predict = []
            for mdl in self.model_list:
                mdl = mdl.replace('\n','')
                base_model = pickle.load(open(mdl,'rb'))
                new_predict = base_model.predict_proba(test_x)
                if(n_dim==1):
                    new_predict = new_predict[:,1:]
                y_predict.append(new_predict)
                
            y_predict = np.hstack(y_predict)
            y_pred = meta_learner.predict_proba(y_predict)
            return y_pred
        
    def predict(self,test_x):
        print('This is \'predict\'. For classification model, please run \'predict_proba\'.')
        
        if(self.ensemble_info['ensemble_method']=='none'):
            mdl0 = self.model_list[0].replace('\n','')
            base_model = pickle.load(open(mdl0,'rb'))
            y_predict = base_model.predict(test_x)
            return y_predict
        
        if(self.ensemble_info['ensemble_method']=='bagging'):
            y_predict = []
            for mdl in self.model_list:
                mdl = mdl.replace('\n','')
                base_model = pickle.load(open(mdl,'rb'))
                y_predict.append(base_model.predict(test_x))
            y_predict = np.array(y_predict)
            return np.average(y_predict,axis=0)
        
        if(self.ensemble_info['ensemble_method']=='ensemble_selection'):
            y_predict = []
            weights = np.array(pd.read_json(self.ensemble_info['ensemble_weights']))[:,0]
            i = 0
            for mdl in self.model_list:
                mdl = mdl.replace('\n','')
                base_model = pickle.load(open(mdl,'rb'))
                y_predict.append(base_model.predict(test_x)*weights[i])
                i+=1
            y_predict = np.array(y_predict)
            return np.sum(y_predict,axis=0)
        
        if(self.ensemble_info['ensemble_method']=='stacking'):
            meta_learner = pickle.load(open(self.ensemble_info['meta_learner_path'],'rb'))
            kfold = self.ensemble_info['kfold']
            mdl0 = self.model_list[0].replace('\n','')
            base_model = pickle.load(open(mdl0,'rb'))
            y_predict = base_model.predict(test_x)
            n_dim = y_predict.shape[1]
            sample_dim = y_predict.shape[0]
            y_predict = []
            if(n_dim==2):
                n_dim = 1
            i=0
            for mdl in self.model_list:
                if(i == 0):
                    new_sumpredict = np.zeros([sample_dim,n_dim])
                mdl = mdl.replace('\n','')
                base_model = pickle.load(open(mdl,'rb'))
                new_predict = base_model.predict(test_x)
                if(n_dim==1):
                    new_predict = new_predict[:,1:]
                new_sumpredict = new_sumpredict + new_predict/kfold
                i+=1
                if(i==kfold):
                    i=0
                    y_predict.append(new_sumpredict)
            
            y_predict = np.hstack(y_predict)
            y_pred = meta_learner.predict(y_predict)
            return y_pred
        
        if(self.ensemble_info['ensemble_method']=='blending'):
            meta_learner = pickle.load(open(self.ensemble_info['meta_learner_path'],'rb'))
            mdl0 = self.model_list[0].replace('\n','')
            base_model = pickle.load(open(mdl0,'rb'))
            y_predict = base_model.predict(test_x)
            n_dim = y_predict.shape[1]
            if(n_dim==2):
                n_dim = 1
            y_predict = []
            for mdl in self.model_list:
                mdl = mdl.replace('\n','')
                base_model = pickle.load(open(mdl,'rb'))
                new_predict = base_model.predict(test_x)
                if(n_dim==1):
                    new_predict = new_predict[:,1:]
                y_predict.append(new_predict)
                
            y_predict = np.hstack(y_predict)
            y_pred = meta_learner.predict(y_predict)
            return y_pred

def save_model(mdl,save_dir):
    mdl_list = ''
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    info = mdl.get_ens_model_info()
    
    
    if(info is None):
        f_ens_info = open(save_dir +'/ens_info','w')
        ens_dict = {}
        ens_dict['ensemble_method'] = 'none'
        f_ens_info.write(json.dumps(ens_dict))
        f_ens_info.close()
        os.system('cp '+ clf.best_algo_path + ' '+save_dir +'/')
        f_mdl_list = open(save_dir +'/model_list','w')
        f_mdl_list.write(os.path.basename(clf.best_algo_path))
        f_mdl_list.close()
        return
        
    f_ens_info = open(save_dir +'/ens_info','w')
    
    ens_dict = {}
    
    if(mdl.task_type == 4):
        ens_dict['task_type'] = 'RGS'
    else:
        ens_dict['task_type'] = 'CLF'
        
    ens_met = info['ensemble_method']
    ens_dict['ensemble_method'] = ens_met
    
    if(ens_met=='bagging'):
        f_ens_info.write(json.dumps(ens_dict))
        
    if(ens_met=='ensemble_selection'):
        ens_dict['ensemble_weights'] = pd.DataFrame(info['ensemble_weights']).to_json()
        f_ens_info.write(json.dumps(ens_dict))
        
    if(ens_met=='stacking'):
        meta_learner_path = save_dir +'/'+os.path.basename(info['meta_learner_path'])
        os.system('cp '+ info['meta_learner_path'] + ' '+save_dir +'/')
        ens_dict['meta_learner_path'] = meta_learner_path
        ens_dict['kfold'] = info['kfold']
        f_ens_info.write(json.dumps(ens_dict))
    
    if(ens_met=='blending'):
        meta_learner_path = save_dir +'/'+os.path.basename(info['meta_learner_path'])
        os.system('cp '+ info['meta_learner_path'] + ' '+save_dir +'/')
        ens_dict['meta_learner_path'] = meta_learner_path
        f_ens_info.write(json.dumps(ens_dict))
        
        
    f_ens_info.close()
    
    if(ens_met=='stacking'):
        for conf in info['config']:
            for partpath in  conf[-1]:
                os.system('cp '+ partpath + ' '+save_dir +'/')
                mdl_list += (os.path.basename(partpath)+'\n')
    else:
        for conf in info['config']:
            os.system('cp '+ conf[-1] + ' '+save_dir +'/')
            mdl_list += (os.path.basename(conf[-1])+'\n')
    f_mdl_list = open(save_dir +'/model_list','w')
    f_mdl_list.write(mdl_list)
    f_mdl_list.close()

def load_model(save_dir):
    
    f_ens_info = open(save_dir +'/ens_info','r')
    ens_info = json.loads(f_ens_info.read())
    f_ens_info.close()
    
    mdl_list = []
    f_mdl_list = open(save_dir +'/model_list','r')
    for mdl in f_mdl_list:
        mdl.replace('\n','')
        mdl_list.append(save_dir +'/'+mdl)
    f_mdl_list.close()
    
    return Ensemble_models(ens_info,mdl_list)