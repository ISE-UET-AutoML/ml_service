import logging

from autogluon.common.features import infer_types
import pandas as pd
from pathlib import Path
import asyncio
from typing import Optional, Union
from datetime import datetime
from autogluon.tabular import TabularPredictor
import pandas as pd
from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config


class AutogluonTrainer(object):
    def __init__(self, kwargs: Optional[dict] = None):
        self.fit_args = None
        self.model_args = None
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.INFO)
        self.data_args = None
        self.has_special_types = None
        self.parse_args(kwargs)

    def parse_args(self, kwargs: Optional[dict] = None):
        if kwargs is None: kwargs = {}
        self.data_args = kwargs.setdefault("data_args", {
            'label': 'label',
            'image_cols': [],
            'text_cols': [],
        })
        self.model_args = kwargs.setdefault("ag_model_args", {
        })
        self.has_special_types = len(self.data_args['image_cols']) + len(self.data_args['text_cols']) > 0
        self.fit_args = kwargs.setdefault("ag_fit_args", {
            "time_limit": 60 * 60 * 60 * 60,
            'presets': 'medium_quality',
            'hyperparameters': get_hyperparameter_config('very_light' if not self.has_special_types else 'multimodal'),
        })

    def train(self, label: str, train_data_path: Path, val_data_path: Path, model_path: Path) -> Union[
        TabularPredictor, None]:
        try:
            train_df = pd.read_csv(train_data_path)
            feature_metadata = 'infer'
            if self.has_special_types:
                from autogluon.tabular import FeatureMetadata
                feature_metadata = FeatureMetadata.from_df(train_df)
                feature_metadata = feature_metadata.add_special_types(
                    {col: ['image_path'] for col in self.data_args['image_cols']})
                feature_metadata = feature_metadata.add_special_types(
                    {col: ['text'] for col in self.data_args['text_cols']})
            # doan nay init lau thi cache lai dung job lib
            # https://joblib.readthedocs.io/en/latest/auto_examples/memory_basic_usage.html
            predictor = TabularPredictor(label=label, path=str(model_path),
                                         **self.model_args)
            # logging.basicConfig(level=logging.DEBUG)
            predictor.fit(
                train_data=str(train_data_path),
                tuning_data=str(val_data_path) if val_data_path is not None else None,
                feature_metadata=feature_metadata,
                **self.fit_args,
            )

            self._logger.info(f"Training completed. Model saved to {model_path}")
            return predictor
        except ValueError as ve:
            self._logger.error(f"Value Error: {ve}")
        except FileNotFoundError as fnfe:
            self._logger.error(f"File Not Found Error: {fnfe}")
        except Exception as e:
            self._logger.error(f"An unexpected error occurred: {e}")
        return None

    async def train_async(self, train_data_path: Path, val_data_path: Path, model_path: Path) -> Union[
        TabularPredictor, None]:
        return await asyncio.to_thread(self.train, train_data_path, val_data_path, model_path)

    @staticmethod
    def evaluate(predictor: TabularPredictor, test_data_path: Path) -> Union[dict, None]:
        try:
            test_data_path = pd.read_csv(test_data_path)
            test_acc = predictor.evaluate(test_data_path)
            return test_acc
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            return None
            # Handle any other unexpected exceptions.
            # It's often a good idea to log the exception details here for later debugging.


if __name__ == "__main__":
    pass
