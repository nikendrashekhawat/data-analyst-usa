from pathlib import Path
import pandas as pd
import numpy as np
import tomllib
from utils.datacollector import KaggleDataCollection
from utils.datacleaner import DataCleaner

    
kdc = KaggleDataCollection()
kdc.get_kaggle_dataset("lukebarousse/data-analyst-job-postings-google-search", "gsearch_jobs.csv")

data_path = kdc.get_data_filepath()

print(data_path)






    
    







# kdc.get_kaggle_dataset("lukebarousse/data-analyst-job-postings-google-search", "gsearch_jobs.csv")

# class DataPreprocessing():
#     pass


# if __name__ == "__main__":
#     pass
