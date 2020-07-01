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
        #ONLY FOR CLF MODEL
        if(self.ensemble_info['task_type'] == 'RGS'):
            print('Regression model does not have \'predict_proba\'.')
            return 'Regression model does not have \'predict_proba\'.'
        
        if(self.ensemble_info['ensemble_method']=='none'):
            femdl = self.model_list[0]
            y_predict = predict_proba_from_path(femdl,test_x)
            return y_predict
        
        if(self.ensemble_info['ensemble_method']=='bagging'):
            y_predict = []
            for femdl in self.model_list:
                y_predict.append(predict_proba_from_path(femdl,test_x))
            y_predict = np.array(y_predict)
            return np.average(y_predict,axis=0)
        
        if(self.ensemble_info['ensemble_method']=='ensemble_selection'):
            y_predict = []
            weights = np.array(pd.read_json(self.ensemble_info['ensemble_weights']))[:,0]
            i = 0
            for femdl in self.model_list:
                y_predict.append(predict_proba_from_path(femdl,test_x)*weights[i])
                i+=1
            y_predict = np.array(y_predict)
            return np.sum(y_predict,axis=0)
        
        if(self.ensemble_info['ensemble_method']=='stacking'):
            meta_learner = pickle.load(open(self.ensemble_info['meta_learner_path'],'rb'))
            kfold = self.ensemble_info['kfold']
            femdl = self.model_list[0]
            y_predict = predict_proba_from_path(femdl,test_x)
            n_dim = y_predict.shape[1]
            sample_dim = y_predict.shape[0]
            y_predict = []
            if(n_dim==2):
                n_dim = 1
            i=0
            for femdl in self.model_list:
                if(i == 0):
                    new_sumpredict = np.zeros([sample_dim,n_dim])
                new_predict = predict_proba_from_path(femdl,test_x)
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
            femdl = self.model_list[0]
            y_predict = predict_proba_from_path(femdl,test_x)
            n_dim = y_predict.shape[1]
            if(n_dim==2):
                n_dim = 1
            y_predict = []
            for femdl in self.model_list:
                new_predict = predict_proba_from_path(femdl,test_x)
                if(n_dim==1):
                    new_predict = new_predict[:,1:]
                y_predict.append(new_predict)
                
            y_predict = np.hstack(y_predict)
            y_pred = meta_learner.predict_proba(y_predict)
            return y_pred

    def predict(self,test_x):

        if(self.ensemble_info['task_type'] == 'CLF'):
            return np.argmax(self.predict_proba(test_x),axis=1)

        if(self.ensemble_info['ensemble_method']=='none'):
            femdl = self.model_list[0]
            y_predict = predict_from_path(femdl,test_x)
            return y_predict
        
        if(self.ensemble_info['ensemble_method']=='bagging'):
            y_predict = []
            for femdl in self.model_list:
                y_predict.append(predict_from_path(femdl,test_x))
            y_predict = np.array(y_predict)
            return np.average(y_predict,axis=0)
        
        if(self.ensemble_info['ensemble_method']=='ensemble_selection'):
            y_predict = []
            weights = np.array(pd.read_json(self.ensemble_info['ensemble_weights']))[:,0]
            i = 0
            for femdl in self.model_list:
                y_predict.append(predict_from_path(femdl,test_x)*weights[i])
                i+=1
            y_predict = np.array(y_predict)
            return np.sum(y_predict,axis=0)
        
        if(self.ensemble_info['ensemble_method']=='stacking'):
            meta_learner = pickle.load(open(self.ensemble_info['meta_learner_path'],'rb'))
            kfold = self.ensemble_info['kfold']
            femdl = self.model_list[0]
            y_predict = predict_from_path(femdl,test_x)
            n_dim = y_predict.shape[1]
            sample_dim = y_predict.shape[0]
            y_predict = []
            if(n_dim==2):
                n_dim = 1
            i=0
            for femdl in self.model_list:
                if(i == 0):
                    new_sumpredict = np.zeros([sample_dim,n_dim])
                new_predict = predict_from_path(femdl,test_x)
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
            femdl = self.model_list[0]
            y_predict = predict_from_path(femdl,test_x)
            n_dim = y_predict.shape[1]
            if(n_dim==2):
                n_dim = 1
            y_predict = []
            for femdl in self.model_list:
                new_predict = predict_from_path(femdl,test_x)
                if(n_dim==1):
                    new_predict = new_predict[:,1:]
                y_predict.append(new_predict)
                
            y_predict = np.hstack(y_predict)
            y_pred = meta_learner.predict(y_predict)
            return y_pred


def save_model(mdl,save_dir):
    mdl_list = ''
    fe_list = ''
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    info = mdl.get_ens_model_info()
    
    
    if(info is None):
        f_ens_info = open(save_dir +'/ens_info','w')
        ens_dict = {}
        ens_dict['ensemble_method'] = 'none'
        f_ens_info.write(json.dumps(ens_dict))
        f_ens_info.close()
        os.system('cp '+ mdl.best_algo_path + ' '+save_dir +'/')
        os.system('cp '+ mdl.best_fe_path + ' '+save_dir +'/')

        f_mdl_list = open(save_dir +'/model_list','w')
        f_mdl_list.write(os.path.basename(mdl.best_algo_path))
        f_mdl_list.close()

        f_fe_list = open(save_dir +'/fe_list','w')
        f_fe_list.write(os.path.basename(mdl.best_fe_path))
        f_fe_list.close()

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
            for partpath in conf[-2]:
                os.system('cp '+ partpath + ' '+save_dir +'/')
                mdl_list += (os.path.basename(partpath)+'\n')
            for partpath in conf[-1]:
            	os.system('cp '+ partpath + ' '+save_dir +'/')
            	fe_list += (os.path.basename(partpath)+'\n')
    else:
        for conf in info['config']:
            os.system('cp '+ conf[-2] + ' '+save_dir +'/')
            os.system('cp '+ conf[-1] + ' '+save_dir +'/')
            mdl_list += (os.path.basename(conf[-2])+'\n')
            fe_list += (os.path.basename(conf[-1])+'\n')
    f_mdl_list = open(save_dir +'/model_list','w')
    f_mdl_list.write(mdl_list)
    f_mdl_list.close()
    f_fe_list = open(save_dir +'/fe_list','w')
    f_fe_list.write(fe_list)
    f_fe_list.close()

def predict_proba_from_path(femdl,test_x):
    fe = femdl[0].replace('\n','')
    base_fe = pickle.load(open(fe,'rb'))
    test_x_tf = base_fe.operate(test_x)
    mdl = femdl[1].replace('\n','')
    base_model = pickle.load(open(mdl,'rb'))
    return base_model.predict_proba(test_x_tf)

def predict_from_path(femdl,test_x):
    fe = femdl[0].replace('\n','')
    base_fe = pickle.load(open(fe,'rb'))
    test_x_tf = base_fe.operate(test_x)
    mdl = femdl[1].replace('\n','')
    base_model = pickle.load(open(mdl,'rb'))
    return base_model.predict(test_x_tf)

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

    fe_list = []
    f_fe_list = open(save_dir +'/fe_list','r')
    for fe in f_fe_list:
        fe.replace('\n','')
        fe_list.append(save_dir +'/'+fe)
    f_fe_list.close()
    
    mdl_list = [[fe_list[i],mdl_list[i]] for i in range(len(mdl_list))]
    
    return Ensemble_models(ens_info,mdl_list)

class bio_models():
    def __init__(self,ensemble_info, mdl_list, imp_ope_list, fb):
        self.ensemble_info = ensemble_info
        self.model_list = mdl_list
        self.fb = fb
        self.imp_ope_list = imp_ope_list
    def predict_proba(self,test_x):
        #ONLY FOR CLF MODEL
        if self.ensemble_info['task_type'] == 'RGS':
            print('Regression model does not have \'predict_proba\'.')
            return 'Regression model does not have \'predict_proba\'.'
        if self.fb is not None:
            test_x = self.fb.transform(test_x)
        y_pred = None
        for key in mdl_list:
            if(np.sum(np.isnan(test_x)) > 0):
                test_x_filled = self.imp_ope_list[key].fit_transform(test_x)
            else:
                test_x_filled = test_x
            if y_pred is None:
                y_pred = mdl[key].predict_proba(test_x_filled)
            else:
                y_pred += mdl[key].predict_proba(test_x_filled)
        y_pred = y_pred/len(mdl_list)
        return y_pred

    def predict(self,test_x):
        if self.ensemble_info['task_type'] == 'CLS':
            return np.argmax(self.predict_proba(test_x),axis=1)
        if self.fb is not None:
            test_x = self.fb.transform(test_x)
        y_pred = None
        for key in mdl_list:
            if(np.sum(np.isnan(test_x)) > 0):
                test_x_filled = self.imp_ope_list[key].fit_transform(test_x)
            else:
                test_x_filled = test_x
            if y_pred is None:
                y_pred = mdl[key].predict(test_x_filled)
            else:
                y_pred += mdl[key].predict(test_x_filled)
        y_pred = y_pred/len(mdl_list)
        return y_pred

def save(biomdl, save_dir, task_type):
    print("PLEASE SAVE THE MODEL IN A NEW FOLDER OR AN EMPTY FOLDER")
    f_ens_info = open(save_dir +'/ens_info','w')
    ens_dict = {}
    ens_dict['task_type'] = task_type
    ens_dict['impute_method'] = biomdl.impute_method
    f_ens_info.write(json.dumps(ens_dict))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if biomdl.fb_operator is not None:
        pkl.dump(biomdl.fb_operator,save_dir+'/fb.pkl')
    for key in biomdl.impute_method:
        pkl.dump(biomdl.impute_operator[key],save_dir+'/'+key+'.pkl')
        save_model(biomdl.mdl[key],save_dir+'/'+key+'_models')


def load(save_dir):
    f_ens_info = open(save_dir +'/ens_info','r')
    ens_info = json.loads(f_ens_info.read())
    f_ens_info.close()
    mdl_list = {}
    imp_ope_list = {}
    if os.path.exists(save_dir+'/fb.pkl'):
        fb = pkl.load(save_dir+'/fb.pkl')
    else:
        fb = None
    for method in ens_info['impute_method']:
        mdl_list[method] = load_model(save_dir+'/'+method+'_models')
        imp_ope_list[method] = pkl.load(save_dir+'/'+method+'.pkl')
    return bio_models(ens_info,mdl_list,imp_ope_list,fb)
