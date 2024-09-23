from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd
import numpy as np
import sklearn
import shap
import lime
from .BaseExplainer import BaseExplainer

class TabularExplainer(BaseExplainer):
    def __init__(self, method="shap", model=None, class_names=None, num_samples=100, sample_data_path=None):
        super().__init__(model)
        self.method = method
        self.class_names = class_names
        self.target_class = None
        self.num_samples = num_samples
        
        if sample_data_path is not None:
            self.sample_data_without_label = pd.read_csv(sample_data_path)
            self.feature_names = self.sample_data.columns
        
        if method == "shap":
            self.explainer = shap.KernelExplainer(self.predict_proba, self.sample_data_without_label)

    def preprocess(self, instance):
        if isinstance(instance, pd.Series):
            processed_input = instance.values.reshape(1,-1)
        if not isinstance(instance, pd.DataFrame):
            processed_input = pd.DataFrame(instance, columns=self.feature_names)
        return processed_input

    def predict_proba(self, instances):
        if isinstance(instances, pd.Series):
            instances = instances.values.reshape(1,-1)
        if not isinstance(instances, pd.DataFrame):
            instances = pd.DataFrame(instances, columns=self.feature_names)
        preds = self.model.predict_proba(instances)
        if self.model.problem_type == "regression" or self.target_class is None:
            return preds
        else:
            return preds[self.target_class] 
    
    # get top 5 features that have the most impact on the prediction of each class
    # return a dictionary with class names as keys and a list of top 5 features as values
    def postprocess(self, instance, k = 5):
        res = []
        
        for i in range(len(self.class_names)):
            feature_importances = instance[:, i]
            top_k_indices = np.argsort(first_elements)[-k:][::-1]
            top_k_column_names = [feature_names[i] for i in top_k_indices if first_elements[i] > 0]
            
            res.append({
                "class": self.class_names[i],
                "features": top_k_column_names
            })
        
        return res
    
    def explain(self, instance):
        processed_input = self.preprocess(instance)
        if self.method == "shap":
            shap_values = self.explainer.shap_values(processed_input, nsamples=self.num_samples)
            return self.postprocess(shap_values[0])
        
        