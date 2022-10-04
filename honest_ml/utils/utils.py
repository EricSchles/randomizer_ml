import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

class RecoverSeeds:
    def __init__(self):
        pass
        
    def mse(self, image_one, image_two):
        error = (
            image_one.astype("float") - image_two.astype("float")
        )
        squared_error = error ** 2
        sum_squared_error = np.sum(
            squared_error
        )
        denominator = image_one.shape[0] * image_two.shape[1]
        return sum_squared_error / float(denominator)
        
    def compare_images(self, image_path_one : str, image_path_two : str):
        processed_image_one = self.preprocess_image(
            image_path_one
        )
        processed_image_two = self.preprocess_image(
            image_path_two
        )

        mse_error = self.mse(
            processed_image_one,
            processed_image_two
        )
        structural_similarity_error = ssim(
            processed_image_one,
            processed_image_two
        )
        return {
            "mse": mse_error,
            "ssim": structural_similarity_error
        }
        
    def preprocess_image(self, image_path: str):
        raw_image = cv2.imread(image_path)
        # greyscale image    
        return cv2.cvtColor(
            raw_image,
            cv2.COLOR_BGR2GRAY
        )
        
