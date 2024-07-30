import abc


class BaseExplainer():
    __metaclass__ = abc.ABCMeta
    def __init__(self, model):
        self.model = model

    @abc.abstractmethod
    def explain(self, example):
        """explain method to be implemented by subclasses"""
        return 
