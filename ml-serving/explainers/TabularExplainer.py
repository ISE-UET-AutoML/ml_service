from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd
import numpy as np
import sklearn
import shap
import lime
from .BaseExplainer import BaseExplainer

class TabularExplainer(BaseExplainer):
    def __init__(self, method="lime", model=None, class_names=None, num_samples=100):
        super().__init__(model)
        self.method = method
        self.class_names = class_names
        self.num_samples = num_samples

    def preprocess(self, instance):
        pass

    def predict_proba(self, instances):
        pass

    def explain(self, instance):
        pass