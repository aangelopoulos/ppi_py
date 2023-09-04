import os, sys
import numpy as np

"""
    datasets.py

    Loads and prepares the datasets for PPI.

    The template is:

    def load_<dataset_name>(path):
        # Load the dataset (and download if needed)
        # Make the dataset into a dataframe
        # Return the dataframe
"""


def load_dataset(dataset_folder, dataset_name, download=True):
    os.makedirs(dataset_folder, exist_ok=True)
    dataset_google_drive_ids = {
        "alphafold": "1lOhdSJEcFbZmcIoqmlLxo3LgLG1KqPho",
        "ballots": "1DJvTWvPM6zQD0V4yGH1O7DL3kfnTE06u",
        "census_income": "15dZeWw-RTw17-MieG4y1ILTZlreJOmBS",
        "census_healthcare": "1RjWsnq-gMngRFRj22DvezcdCVl2MxAIX",
        "forest": "1Vqi1wSmVnWh_2lLQuDwrhkGcipvoWBc0",
        "galaxies": "1pDLQesPhbH5fSZW1m4aWC-wnJWnp1rGV",
        "gene_expression": "17PwlvAAKeBYGLXPz9L2LVnNJ66XjuyZd",
        "plankton": "1KEk0ZFZ6KiB7_2tdPc5fyBDFNhhJUS_W",
    }
    if dataset_name not in dataset_google_drive_ids.keys():
        raise NotImplementedError(
            f"The dataset {dataset_name} is not implemented. Valid options are {list(dataset_google_drive_ids.keys())}."
        )
    dataset_path = os.path.join(dataset_folder, dataset_name + ".npz")
    if not os.path.exists(dataset_path):
        print(
            f"Dataset {dataset_name} not found at location {dataset_folder}; downloading now..."
        )
        os.system(
            f"gdown {dataset_google_drive_ids[dataset_name]} -O {dataset_path}"
        )
    return np.load(dataset_path)
