import numpy as np
import os
import pandas as pd
from imgaug import augmenters as iaa
import shutil

def get_sample_counts(output_directory: str, datasets: str, class_names: list):
    """
    Class-wise positive sample count of a dataset

    
    Arguments:
        output_directory {str} -- folder containing the dataset.csv file
        datasets {str} -- train|validation|test set(s)
        class_names {list of str} -- target classes 
    """

    df = pd.read_csv(os.path.join(output_directory, f"{datasets}.csv"))
    total_count = df.shape[0]
    labels = df[class_names].as_matrix()
    positive_counts = np.sum(labels, axis=0)
    class_positive_counts = dict(zip(class_names, positive_counts))
    return total_count, class_positive_counts


def get_class_weights(total_counts: int, class_positive_counts: dict, multiply: int):
    """Calculate the class_weight used in training

    Arguments:
        total_counts {int} -- total counts (name implies)
        class_positive_counts {dict} -- dict of int, eg. {"Effusion": 300, "Infiltration": 300}
        multiply {int} -- positive weight multiply
    """

    def get_single_class_weight(pos_counts, total_counts):
        denominator = (total_counts - pos_counts) * multiply + pos_counts
        return {
            0: pos_counts / denominator,
            1: (denominator - pos_counts) / denominator,
        }
    
    class_names = list(class_positive_counts.keys())
    label_counts = np.array(list(class_positive_counts.values()))
    class_weights = []
    for i, class_name in enumerate(class_names):
        class_weights.append(get_single_class_weight(label_counts[i], total_counts))

    return class_weights


def augmenter():
    """
    Method to augment images.

    Following from CheXNet paper, images were randomly flipped with 50% probability

    """
    augmenter = iaa.Sequential(
        [
            iaa.Fliplr(0.5),
        ],
        random_order=True
    )
    return augmenter

def check_create_output_dir(output_directory: str):
    """
    Checks for and creates (if non-existent) directory for each experiment
    
    Arguments:
        output_dir {str} -- Where on the filesystem to save the experiment
    """

    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)

    # check used to verify if this directly is being utilized
    running_flag_file = os.path.join(output_directory, ".training.lock")
    if os.path.isfile(running_flag_file):
        raise RuntimeError("There is a process currently utilizing this directory!")
    else:
        create_training_lock(output_directory)
        return True
    return False

def create_training_lock(output_directory: str):
    """
    Create training lock for the directory where an experiment is potentially running
    
    Arguments:
        output_directory {str} -- directory where experiment is currently executing
    """
    running_flag_file = os.path.join(output_directory, ".training.lock")
    open(running_flag_file, "a").close()
    

def delete_training_lock(output_directory: str):
    """
    Remove a potential .training.lock file on a directly where an experiment is/has been run
    
    Arguments:
        output_directory {str} -- directory where an experiment has or is running
    """
    running_flag_file = os.path.join(output_directory, ".training.lock")
    return os.remove(running_flag_file)


def backup_config_file(output_directory: str, config_file: str):
    """
    Backup a copy of the current configuration file to 
    experiment directory defined in configuration file
    
    Arguments:
        output_directory {str} -- Where on the filesystem to save the backup file
        config_file {str} -- Filename and location of current experiment configuration file
    """

    try:
        print(f"Backing up configuration file to {output_directory}")
        shutil.copy(config_file, os.path.join(output_directory, os.path.split(config_file)[1]))
    except:
        raise RuntimeError("Unable to save experiment configuration file! Please remedy this problem before proceeding.")

def build_datasets(dataset_csv_dir: str, output_directory: str):
    """
    Import and define partition datasets according to provided .csv file
    
    Arguments:
        dataset_csv_dir {str} -- directory where dataset CSV is stored
        output_directory {str} -- Current experiment directory

    """

    datasets = ["train", "validation", "test"]
    for dataset in datasets:
        shutil.copy(os.path.join(dataset_csv_dir, f"{dataset}.csv"), output_directory)
    
    return

