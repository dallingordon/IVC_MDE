## DO NOT CHANGE THIS FILE
## Your changes will be discarded at the server evaluation

import os
import numpy as np
from PIL import Image

from my_submission.user_config import MyDepthModel

class AIcrowdWrapper:
    """
        Entrypoint for the evaluator to connect to the user's agent
        Abstracts some operations that are done on client side
            - Reading images from shared disk
            - Checking predictions for basic issues
            - Writing predictions to shared disk
    """
    def __init__(self,
                 dataset_dir='./public_dataset/images/',
                 predictions_dir='./evaluator_outputs/'):
                 
        self.model = MyDepthModel()
        shared_dir = os.getenv("AICROWD_PUBLIC_SHARED_DIR", None)
        if shared_dir is not None:
            self.predictions_dir = os.path.join(shared_dir, 'predictions')
        else:
            self.predictions_dir = predictions_dir
        assert os.path.exists(self.predictions_dir), f'{self.predictions_dir} - No such directory'
        self.dataset_dir = os.getenv("AICROWD_DATASET_DIR", dataset_dir)
        assert os.path.exists(self.dataset_dir), f'{self.dataset_dir} - No such directory'

    def raise_aicrowd_error(self, msg):
        """ Will be used by the evaluator to provide logs """
        raise NameError(msg)

    @staticmethod
    def read_image(path):
        image = np.array(Image.open(path))
        return image

    @staticmethod
    def check_output(prediction):
        assert ~np.any(np.isnan(prediction)), "Prediction shouln't contain nan values"
    
    def save_prediction(self, filename, prediction):
        pred_scaled = np.int32(prediction * 128 + 1)
        img = Image.fromarray(pred_scaled)
        img.save(filename)

    def predict_depth_single_image(self, filename):
        image_path = os.path.join(self.dataset_dir, filename)
        image = self.read_image(image_path)
        
        prediction = self.model.predict_depth_single_image(image)

        self.check_output(prediction)

        prediction_path = os.path.join(self.predictions_dir, filename)
        self.save_prediction(prediction_path, prediction)

        return True
