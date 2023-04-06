import numpy as np

class RandomDepthPredictor:
    def __init__(self):
        """
        Initialize your model here
        """
        pass

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

        image_size = image_to_predict_depth.shape[:2]
        depth_image = np.random.rand(*image_size) * 20 + 1
        return depth_image