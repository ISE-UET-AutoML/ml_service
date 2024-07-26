import pandas as pd
import os
import warnings
import numpy as np
from matplotlib.image import imread

from autogluon.multimodal import MultiModalPredictor
from PIL import Image
from lime import lime_image
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries

# supported methods: LIME, SHAP

class ImageExplainer():
    def __init__(self, method="lime", predictor=None, temp_path=None, num_samples=5):
        self.method = method
        self.predictor = predictor
        self.temp_path = temp_path
        self.num_samples = num_samples

        if self.method ==  "lime":
            self.explainer = lime_image.LimeImageExplainer()


    
    def predict_proba(self, image_data):
        proba_list = []
        data = pd.DataFrame(columns=["image"])
        for i in range(image_data.shape[0]):
            img = Image.fromarray(image_data[i], 'RGB')
            img_path = f"{self.temp_path}/{i}.jpg"
            img.save(img_path)
            data = data._append({"image": img_path}, ignore_index=True)

        proba_list = self.predictor.predict_proba(data, as_multiclass=True)
        return np.asarray(proba_list).reshape(image_data.shape[0], 4)
    

    def explain(self, image_path, image_explain_path):
        print('Explaining')
        image = imread(image_path)
        try:
            explanation = self.explainer.explain_instance(image, self.predict_proba, hide_color=0, num_samples=self.num_samples, batch_size=500)
            temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=10, hide_rest=False)

            # Display the explanation
            marked_image = mark_boundaries(temp, mask)
            plt.imsave(image_explain_path, marked_image)
        except Exception as e:
            print(e)
            print("Error in explaining image")
        return image_explain_path
    