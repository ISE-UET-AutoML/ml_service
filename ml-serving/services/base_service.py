import onnxruntime as ort
import typing as t
import abc


class BaseService():
    __metaclass__ = abc.ABCMeta
    def __init__(self):
        print("Init")

    # each task has its own fake data: images -> list of fake image paths, ....
    def get_fake_data(self, task: str) -> t.Any:
        # if task == "image_classification":
        #     return ["../ml-serving/sample_data/image_classify.jpg"]
        # elif task == "text_classification":
        #     return ["../ml-serving/sample_data/text_classify.txt"]
        # else:
        #     return None
        pass



    def deploy(self, **params: t.Any) -> dict:
        print(params)
        userEmail = params["userEmail"]
        projectName = params["projectName"]
        runName = params["runName"]
        task = params["task"]
        try:
            self.load_model_and_model_info(userEmail, projectName, runName)
            self.warmup(task)
        except Exception as e:
            print(e)
            print("Model deploy failed")
            return {"status": "failed", "message": "Model deploy failed"}
        return {"status": "success", "message": "Model deploy successful"}
        
    
    def check_already_deploy(self, **params: t.Any) -> dict:
        if hasattr(self, 'ort_sess'):
            return {"status": "success", "message": "Model already deployed"}
        else:
            self.deploy(**params)
            return {"status": "success", "message": "Model deployed successfully"}
        
    # FIX RELATIVE PATH ERROR
    def load_model_and_model_info(self, userEmail: str, projectName: str, runName: str) -> t.Tuple[t.Any, t.List[str]]:
        try:
            self.ort_sess = ort.InferenceSession(f'../tmp/{userEmail}/{projectName}/trained_models/ISE/{runName}/model.onnx', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            self.input_names = [in_param.name for in_param in self.ort_sess.get_inputs()]
            print("Model deploy successfully")
            return None
        except Exception as e:
            print(e)

    # fake data and create requests to warmup the model
    def warmup(self, task):
        # data = self.get_fake_data(task)
        # self.predict_proba(data)
        pass


    @abc.abstractmethod
    def predict_proba(self, data):
        """method to be implemented by subclasses"""
        return


    @abc.abstractmethod
    def predict(self):
        """predict method to be implemented by subclasses"""
        return

    @abc.abstractmethod
    def explain(self):
        """explain method to be implemented by subclasses"""
        return



