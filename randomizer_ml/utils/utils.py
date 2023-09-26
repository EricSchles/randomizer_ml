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
        
class Sampler:
    def __init__(self):
        pass

    def rescale(self, data, new_max):
        old_max = data.max()
        new_data = (new_max/old_max) * (data - old_max) + new_max
        return new_data.astype(int)

    def sample(self, data, sample_size, seed=1, method="choice"):
        np.random.seed(seed)
        index = list(range(len(data)))
        if method == "choice":
            new_index = np.random.choice(
                index, size=sample_size
            )
        if method == "exponential":
            new_index = np.random.exponential(
                scale=data.shape[0]//2, 
                size=sample_size
            )
            new_index = self.rescale(
                new_index, data.shape[0]-1
            )
        if method == "gamma":
            shape = np.random.choice(
                [0.5, 3, 5]
            )
            new_index = np.random.gamma(
                shape=shape,
                scale=data.shape[0]//2, 
                size=sample_size
            )
            new_index = self.rescale(
                new_index, data.shape[0]-1
            )
        if method == "normal":
            new_index = np.random.normal(
                loc=data.shape[0]//2,
                scale=data.shape[0]//4,
                size=sample_size
            )
            new_index = self.rescale(
                new_index, data.shape[0]-1
            )
        if method == "poisson":
            new_index = np.random.poisson(
                lam=4,
                size=sample_size
            )
            new_index = self.rescale(
                new_index, data.shape[0]-1
            )
        if isinstance(data, np.ndarray):
            data = pd.Series(data)
        return data.iloc[new_index]
