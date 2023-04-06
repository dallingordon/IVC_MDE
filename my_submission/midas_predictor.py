import numpy as np
import torch

import sys
sys.path.append('./my_submission/MiDaS/')

from midas.model_loader import load_model
from run import process

class MiDaSPredictor:
    def __init__(self):
        """
        Initialize your model here
        """
        model_path = './my_submission/models/dpt_large_384.pt'
        self.model_type = 'dpt_large_384'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.transform, self.net_w, self.net_h = load_model(self.device, model_path, self.model_type, 
                        optimize=False, height=None, square=False)

    def raise_aicrowd_error(self, msg):
        """ Will be used by the evaluator to provide logs, DO NOT CHANGE """
        raise NameError(msg)
    
    def predict_depth_single_image(self, image_to_predict_depth):
        """
        Implements the ranking function for a given instruction
        Inputs:
            image_to_predict_depth - Single frame from onboard the flight

        Outputs:
            An 2D image with pixel values being the corresponding depth
        """

        # MiDaS Compatible image load - 0 to 1 scaling
        midas_input_rgb = image_to_predict_depth/256 
        if midas_input_rgb.ndim == 2:
            midas_input_rgb = np.stack((midas_input_rgb,)*3, axis=-1)
        with torch.no_grad():
            t_image = self.transform({"image": midas_input_rgb})['image']
            disparity_image = process(self.device, self.model, self.model_type, 
                                  t_image, (self.net_w, self.net_h), 
                                  image_to_predict_depth.shape[1::-1],
                                  False, False)
        
        # convert disparty output of midas to depth image
        depth_image = disparity_image.max() / (disparity_image + 1e-8) 
        depth_image = np.clip(depth_image, 1, 1000)

        return depth_image