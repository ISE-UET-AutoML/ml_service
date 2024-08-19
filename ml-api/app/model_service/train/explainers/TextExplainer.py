from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd
import numpy as np
import sklearn
import shap
import lime
from .BaseExplainer import BaseExplainer
from transformers import AutoTokenizer


class TextExplainer(BaseExplainer):
    def __init__(self, method="shap", model=None, class_names=None):
        super().__init__(model, class_names)
        self.method = method
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

        # TODO: fix hardcode class names
        if self.class_names == [0, 1]:
            self.class_names = ["negative", "positive"]
        if method == "shap":
            self.explainer = shap.Explainer(self.predict_proba, self.tokenizer, output_names=self.class_names)


    def preprocess(self, instance):
        if self.method == "shap":
            return pd.DataFrame([instance], columns=['text'])

    def predict_proba(self, instances):
        sentences = []
        for instance in instances:
            sentences.append(instance)
        return self.model.predict_proba({'text': sentences}, realtime=True)

    def explain(self, instance):
        data = self.preprocess(instance)
        if self.method == "shap":
            shap_values = self.explainer(data['text'][0:1], max_evals=100, batch_size=20)
            return shap.plots.text(shap_values, display=False)
        else:
            return "Method not supported"