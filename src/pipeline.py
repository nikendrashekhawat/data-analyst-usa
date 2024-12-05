from pathlib import Path
import pandas as pd
import numpy as np
import tomllib
from data_collection import KaggleDataCollection

def is_directory_empty(dir_path):
    # Check if the directory is empty
    return not any(Path(dir_path).iterdir())

with open("configs/cfg.toml", "rb") as cfg:
    configs = tomllib.load(cfg)
    
KAGGLE_USERNAME = configs["kaggle"]["username"]
KAGGLE_APIKEY = configs["kaggle"]["key"]  
    
kdc = KaggleDataCollection(KAGGLE_USERNAME, KAGGLE_APIKEY)
# kdc.get_kaggle_dataset("lukebarousse/data-analyst-job-postings-google-search", "gsearch_jobs.csv")

cwd = Path(".").resolve()
data_path = Path("dataset/gsearch_jobs.csv")

if not (cwd/data_path).is_file():
    raise FileExistsError(f"{data_path} does not exists.")






    
    







# kdc.get_kaggle_dataset("lukebarousse/data-analyst-job-postings-google-search", "gsearch_jobs.csv")

# class DataPreprocessing():
#     pass


# if __name__ == "__main__":
#     pass
