# -*- coding: utf-8 -*-
"""
utilities.py
~~~~~~~~~~~~

Utility functions for automl service.
"""
import copy
import json

import pandas as pd
import yaml
import sklearn


#TODO: change name to load json
def read_params(d):
    """Parse parameter file"""
    return yaml.load(d)

def cross_validate(cl, X_train, y_train):
    cv = sklearn.model_selection.cross_validate(cl, X_train, y_train,
                                                cv=5, scoring=scoring)
    mean_accuracy = cv['test_accuracy'].mean()
    mean_roc_auc = cv['test_roc_auc'].mean()
    return (mean_accuracy, mean_roc_auc)

class ModelFactory(object):

    def __init__(self):
        self.pipelines = dict()

    def __getitem__(self, item):
        return self.pipelines[item]

    def add_pipeline(self, mdl, train_data, _id):
        pipeline_id = _id
        self.pipelines[pipeline_id] = dict()
        pipeline = self.pipelines[pipeline_id]
        pipeline['data'] = train_data
        pipeline['model'] = mdl

        

