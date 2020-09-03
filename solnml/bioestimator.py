import argparse
import os
import sys
import time
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris,load_breast_cancer
from sklearn.metrics import balanced_accuracy_score, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.append(os.getcwd())
from solnml.utils.data_manager import DataManager
from solnml.estimators import Classifier as soln_Classifier
from solnml.estimators import Regressor as soln_Regressor
from solnml.utils import saveloadmodel
from solnml.automl import AutoML
from solnml.components.metrics.metric import get_metric
from solnml.components.feature_engineering.transformation_graph import DataNode
from featureband.feature_band import FeatureBand
from featureband.util.data_util import load_dataset
from featureband.util.metrics_util import evaluate_cross_validation, load_clf
from fancyimpute import KNN, IterativeSVD, MatrixFactorization, SimpleFill
#tensorflow == 1.13.0rc1
#keras == 2.2.4
class Classifier():
    def __init__(
            self,
            dataset_name='default_dataset_name',
            time_limit=10800,
            amount_of_resource=None,
            metric='acc',
            include_algorithms=None,
            enable_meta_algorithm_selection=True,
            ensemble_method='ensemble_selection',
            ensemble_size=50,
            per_run_time_limit=150,
            random_state=1,
            n_jobs=1,
            evaluation='holdout',
            impute = False,
            impute_method = ['MatrixFactorization','KNN','IterativeSVD'],
            pre_fs = True,
            fb_k = 300,
            fb_r0 = 50,
            fb_max_iter = 50,
            fb_population_size = 10,
            fb_n0 = 500,
            fb_metric = 'accuracy',
            output_dir="/tmp/"):
        self.dataset_name = dataset_name
        self.metric = metric
        self.task_type = None
        self.time_limit = time_limit
        self.amount_of_resource = amount_of_resource
        self.include_algorithms = include_algorithms
        self.enable_meta_algorithm_selection = enable_meta_algorithm_selection
        self.ensemble_method = ensemble_method
        self.ensemble_size = ensemble_size
        self.per_run_time_limit = per_run_time_limit
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.evaluation = evaluation
        self.output_dir = output_dir
        self.dm = None
        #imputation arguments
        self.impute = impute
        self.impute_method = impute_method
        self.impute_operator = {}
        #fb arguments
        self.pre_fs = pre_fs
        self.fb_k = fb_k
        self.fb_r0 = fb_r0
        self.fb_max_iter = fb_max_iter
        self.fb_population_size = fb_population_size
        self.fb_n0 = fb_n0
        self.fb_metric = fb_metric #['accuracy','f1_score']
        self.origin_X = None
        self.origin_y = None
        self.fb_operator = None
        self.headers = None
        self.selected_headers = None
        self.train_data = {}
        self.test_data = {}
        self.mdl = {}
        self.y_pred = {}
        # Create output directory.
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not self.impute:
            self.impute_method = ['mean'] #simply impute anyway

    def run_impute(self, X, state = 'train'):
        if state == 'train':
            self.train_data['ave'] = np.zeros([X.shape[0],X.shape[1]])
            for imp_method in self.impute_method:
                print('Impute '+imp_method+' starts!')
                if imp_method == 'mean':
                    imp_ope = SimpleFill()
                if imp_method == 'KNN':
                    imp_ope = KNN()
                if imp_method == 'IterativeSVD':
                    imp_ope = IterativeSVD()
                if imp_method == 'MatrixFactorization':
                    imp_ope = MatrixFactorization()
                X_filled = imp_ope.fit_transform(X)
                self.train_data[imp_method] = X_filled
                self.impute_operator[imp_method] = imp_ope
                self.train_data['ave'] += X_filled
                print('Impute '+imp_method+' ends!')
            self.train_data['ave'] /= len(self.impute_method)
        return 0

    def feature_selection(self, X, y, state = 'train'):
        if state == 'train':
            print('Feature_selection starts!')
            fb = FeatureBand(r0 = self.fb_r0, 
                n0 = self.fb_n0, 
                clf = load_clf('logistic'), 
                max_iter = self.fb_max_iter, 
                k = self.fb_k, 
                population_size = self.fb_population_size, 
                local_search = True)
            times, iter_best, global_best = fb.fit(X, y, metrics=self.fb_metric)
            for key in self.train_data:
                self.train_data[key] = fb.transform(self.train_data[key])
            self.fb_operator = fb
            if self.headers is not None:
                self.selected_headers = self.headers[fb.featrue_selected]
        return 0

    def preprocess(self, X, y):

        self.run_impute(X)
        if X.shape[1] > self.fb_k and self.pre_fs == True:
            self.feature_selection(self.train_data['ave'], y)
        return 0



    def fit(self, X, y):
        if isinstance(X,pd.DataFrame):
            self.headers = X.columns
        X = np.array(X)
        y = np.array(y)
        self.origin_X = X
        self.origin_y = y

        if(np.sum(np.isnan(X)) == 0):
            self.impute = False
            self.impute_method = ['mean']

        self.preprocess(X, y)
        for key in self.impute_operator:
            X_train = self.train_data[key]
            self.dm = DataManager(X_train, y)
            train_data = self.dm.get_data_node(X_train, y)
            self.mdl[key] = soln_Classifier(time_limit=self.time_limit/len(self.impute_method),
                output_dir=self.output_dir,
                ensemble_method=self.ensemble_method,
                evaluation=self.evaluation,
                metric=self.metric,
                n_jobs=self.n_jobs)
            self.mdl[key].fit(train_data)
        return 0

    def predict_proba(self, X_test):
        y_pred = None
        X_test = np.array(X_test)
        for key in self.impute_operator:
            if np.sum(np.isnan(X_test)) > 0:
                X_test_filled = self.impute_operator[key].fit_transform(X_test)
            else:
                X_test_filled = X_test
            if X_test.shape[1] > self.fb_k and self.pre_fs == True:
                X_test_filled = self.fb_operator.transform(X_test_filled)
            test_data = self.dm.get_data_node(X_test_filled, [])
            self.y_pred[key] = self.mdl[key].predict_proba(test_data)
            if y_pred is None:
                y_pred = self.y_pred[key]
            else:
                y_pred += self.y_pred[key]
        y_pred /= len(self.impute_operator)
        return y_pred

    def predict(self, X_test):
        return np.argmax(self.predict_proba(X_test),axis=1)

    @property
    def get_feature_selected(self):
        if self.fb_operator is not None:
            return self.fb_operator.featrue_selected

    def feature_analysis(self,topk = 30):
        from lightgbm import LGBMClassifier
        relation_list = {}
        result = None
        importance_array = []
        for key in self.impute_operator:
            mdl = self.mdl[key]
            data = self.dm.get_data_node(self.train_data[key], self.origin_y)
            data_tf = mdl.data_transform(data).data[0]
            sub_topk = min(int(topk/len(self.impute_operator)),data_tf.shape[1])
            lgb = LGBMClassifier()
            lgb.fit(data_tf,self.origin_y)
            _importance = lgb.feature_importances_
            index_needed = np.argsort(-_importance)[:sub_topk]

            temp_array = _importance[index_needed]
            temp_array = temp_array/np.max(temp_array)
            importance_array.append(temp_array)

            relation_list[key] = mdl.feature_corelation(data, index_needed)
            relation_list[key].index = key+relation_list[key].index
            if self.selected_headers is not None:
                relation_list[key].columns = list(self.selected_headers)
            elif self.fb_operator is not None:
                relation_list[key].columns = ['origin_fearure' + str(it) for it in self.fb_operator.featrue_selected]
            if result is None:
                result = relation_list[key]
            else:
                result = result.append(relation_list[key])

        importance_frame = pd.DataFrame(np.hstack(importance_array))
        importance_frame.columns = ['feature_importance']
        importance_frame.index = result.index
        return pd.concat([importance_frame,result],axis=1)
        return result

    def save(self,save_dir):
        saveloadmodel.save(self,save_dir,task_type = 'CLF')






class Regression_selector():
    def __init__(self, k):
        self.k = k
        self.featrue_selected = None
    def fit(self, X, y):
        from lightgbm import LGBMRegressor
        lgb = LGBMRegressor(random_state=1)
        lgb.fit(X, y)
        _importance = lgb.feature_importances_
        self.featrue_selected = np.argsort(-_importance)[:k]
    def transform(X):
        return X[:,self.featrue_selected]



class Regressor():
    def __init__(self,
            dataset_name='default_dataset_name',
            time_limit=10800,
            amount_of_resource=None,
            metric='acc',
            include_algorithms=None,
            enable_meta_algorithm_selection=True,
            ensemble_method='ensemble_selection',
            ensemble_size=50,
            per_run_time_limit=150,
            random_state=1,
            n_jobs=1,
            evaluation='holdout',
            impute = False,
            impute_method = ['KNN','IterativeImputer','MatrixFactorization'],
            fb_k = 200,
            fb_r0 = 50,
            fb_max_iter = 50,
            fb_population_size = 10,
            fb_n0 = 500,
            output_dir="/tmp/"):
        self.dataset_name = dataset_name
        self.metric = metric
        self.task_type = None
        self.time_limit = time_limit
        self.amount_of_resource = amount_of_resource
        self.include_algorithms = include_algorithms
        self.enable_meta_algorithm_selection = enable_meta_algorithm_selection
        self.ensemble_method = ensemble_method
        self.ensemble_size = ensemble_size
        self.per_run_time_limit = per_run_time_limit
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.evaluation = evaluation
        self.output_dir = output_dir
        self.dm = None
        #imputation arguments
        self.impute = impute
        self.impute_method = impute_method
        self.impute_operator = {}
        #fb arguments
        self.fb_k = fb_k
        self.origin_X = None
        self.origin_y = None
        self.fb_operator = None
        self.headers = None
        self.selected_headers = None
        self.train_data = {}
        self.test_data = {}
        self.mdl = {}
        self.y_pred = {}
        # Create output directory.
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not self.impute:
            self.impute_method = ['mean'] #simply impute anyway

    def run_impute(self, X, state = 'train'):
        if state == 'train':
            self.train_data['ave'] = np.zeros([X.shape[0],X.shape[1]])
            for imp_method in self.impute_method:
                if imp_method == 'mean':
                    imp_ope = SimpleFill()
                if imp_method == 'KNN':
                    imp_ope = KNN()
                if imp_method == 'IterativeSVD':
                    imp_ope = IterativeSVD()
                if imp_method == 'MatrixFactorization':
                    imp_ope = MatrixFactorization()
                X_filled = imp_ope.fit_transform(X)
                self.train_data[imp_method] = X_filled
                self.impute_operator[imp_method] = imp_ope
                self.train_data['ave'] += X_filled
            self.train_data['ave'] /= len(self.impute_method)
        return 0

    def feature_selection(self, X, y, state = 'train'):
        if state == 'train':
            fb = Regression_selector(self.fb_k)
            fb.fit(X, y)
            for key in self.train_data:
                self.train_data[key] = fb.transform(self.train_data[key])
            self.fb_operator = fb
            if self.selected_headers is not None:
                self.selected_headers = self.headers[fb.featrue_selected]
        return 0

    def preprocess(self, X, y):

        self.run_impute(X)
        if X.shape[1] > self.fb_k and self.pre_fs == True:
            self.feature_selection(self.train_data['ave'], y)
        return 0



    def fit(self, X, y):
        if isinstance(X,pd.DataFrame):
            self.headers = X.columns
        X = np.array(X)
        y = np.array(y)
        self.origin_X = X
        self.origin_y = y

        if(np.sum(np.isnan(X)) == 0):
            self.impute = False
            self.impute_method = ['mean']

        self.preprocess(X, y)
        for key in self.impute_operator:
            X_train = self.train_data[key]
            self.dm = DataManager(X_train, y)
            train_data = self.dm.get_data_node(X_train, y)
            self.mdl[key] = soln_Regressor(time_limit=self.time_limit/len(self.impute_method),
                output_dir=self.output_dir,
                ensemble_method=self.ensemble_method,
                evaluation=self.evaluation,
                metric=self.metric,
                n_jobs=self.n_jobs)
            self.mdl[key].fit(train_data)
        return 0

    def predict(self, X_test):
        X_test = np.array(X_test)
        y_pred = None
        for key in self.impute_operator:
            if(np.sum(np.isnan(X_test)) > 0):
                X_test_filled = self.impute_operator[key].fit_transform(X_test)
            else:
                X_test_filled = X_test
            if X_test.shape[1] > self.fb_k and self.pre_fs == True:
                X_test_filled = self.fb_operator.transform(X_test_filled)
            test_data = self.dm.get_data_node(X_test_filled, [])
            self.y_pred[key] = self.mdl[key].predict(test_data)
            if y_pred is None:
                y_pred = self.y_pred[key]
            else:
                y_pred += self.y_pred[key]
        y_pred /= len(self.impute_operator)
        return y_pred

    @property
    def get_feature_selected(self):
        if self.fb_operator is not None:
            return self.fb_operator.featrue_selected


    def feature_analysis(self,topk = 30):
        from lightgbm import LGBMRegressor
        relation_list = {}
        result = None
        importance_array = []
        for key in self.impute_operator:
            mdl = self.mdl[key]
            data = self.dm.get_data_node(self.train_data[key], self.origin_y)
            data_tf = mdl.data_transform(data).data[0]
            sub_topk = min(int(topk/len(self.impute_operator)),data_tf.shape[1])
            lgb = LGBMRegressor()
            lgb.fit(data_tf,self.origin_y)
            _importance = lgb.feature_importances_
            index_needed = np.argsort(-_importance)[:sub_topk]

            temp_array = _importance[index_needed]
            temp_array = temp_array/np.max(temp_array)
            importance_array.append(temp_array)

            relation_list[key] = mdl.feature_corelation(data, index_needed)
            relation_list[key].index = key+relation_list[key].index
            if self.selected_headers is not None:
                relation_list[key].columns = list(self.selected_headers)
            elif self.fb_operator is not None:
                relation_list[key].columns = ['origin_fearure' + str(it) for it in self.fb_operator.featrue_selected]
            if result is None:
                result = relation_list[key]
            else:
                result = result.append(relation_list[key])

        importance_frame = pd.DataFrame(np.hstack(importance_array))
        importance_frame.columns = ['feature_importance']
        importance_frame.index = result.index
        return pd.concat([importance_frame,result],axis=1)
        return result
        
    def save(self,save_dir):
        saveloadmodel.save(self,save_dir,task_type = 'RGS')

