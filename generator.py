import numpy as np
import os
import pandas as pd
from keras.utils import Sequence
from PIL import Image
from skimage.transform import resize

class AugmentedImageSequence(Sequence):
    """
    Class for generating augmented image sequences
    
    Arguments:
        Sequence {[type]} -- [description]
    """

    def __init__(self, dataset_csv_file: str, class_names: list, source_image_dir: str,
                batch_size=16, target_size=(224, 224), augmenter=None,
                verbose=0, steps=None, shuffle_on_epoch_end=True, random_state=1):
                """                 
                Arguments:
                    dataset_csv_file {str} -- Path of dataset CSV (assuming CSV here)
                    class_names {list} -- List of class names
                    source_image_dir {str} -- Path of source imagees
                
                Keyword Arguments:
                    batch_size {int} -- [description] (default: {16})
                    target_size {tuple} -- [description] (default: {(224, 224)})
                    augmenter {imgaug} -- [description] (default: {None})
                    verbose {int} -- [description] (default: {0})
                    steps {int or str} -- [description] (default: {None})
                    shuffle_on_epoch_end {bool} -- [description] (default: {True})
                    random_state {int} -- [description] (default: {1})
                """

                self.dataset_dataframe = pd.read_csv(dataset_csv_file)
                self.source_image_dir = source_image_dir
                self.batch_size = batch_size
                self.target_size = target_size
                self.augmenter = augmenter
                self.verbose = verbose
                self.shuffle = shuffle_on_epoch_end
                self.random_state = random_state
                self.class_names = class_names
                self.prepare_dataset()
                if steps is None:
                    self.steps = int(np.ceil(len(self.x_path) / float(self.batch_size)))
                else:
                    self.steps = int(steps)

                def __bool__(self):
                    return True
                
                def __len__(self):
                    return self.steps
                
                def __getitem__(self, idx):
                    batch_x_path = self.x_path[idx * self.batch_size:(idx + 1) * self.batch_size]
                    batch_x = np.asarray([self.load_image(x_path) for x_path in batch_x_path])
                    batch_x = self.transform_batch_images(batch_x)
                    batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
                    return batch_x, batch_y
                
                def load_image(self, image_file):
                    image_path = os.path.join(self.source_image_dir, image_file)
                    image = Image.open(image_path)
                    image_array = np.asarray(image.convert("RGB"))
                    image_array = image_array / 255.
                    image_array = resize(image_array, self.target_size)
                    return image_array
                
                def transform_batch_images(self, batch_x):
                    """
                    Normalize the batch and center the data accordingly
                    
                    Arguments:
                        batch_x {} -- batch
                    
                    Returns:
                        [type] -- [description]
                    """

                    if self.augmenter is not None:
                        batch_x = self.augmenter.augment_images(batch_x)
                    imagenet_mean = np.array([0.485, 0.456, 0.406])
                    imagenet_stddev = np.array([0.229, 0.224, 0.225])
                    batch_x = (batch_x - imagenet_mean) / imagenet_stddev
                    return batch_x
                
                def get_y_true(self):
                    if self.shuffle:
                        raise ValueError("""
                        You're trying to run get_y_true() when generator option 'shuffle_on_epoch_end is True.
                        """)
                    return self.y[:self.steps*self.batch_size, :]
                
                def prepare_dataset(self):
                    df = self.dataset_df.sample(frac=1., random_state=self.random_state)
                    self.x_path, self.y = df["Image Index"].as_matrix(), df[self.class_names].as_matrix()

                def on_epoch_end(self):
                    if self.shuffle:
                        self.random_state += 1
                        self.prepare_dataset()

                