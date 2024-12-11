from typing import Optional
import os
import pandas as pd
import numpy as np
import tomllib
from utils.datacollector import KaggleDataCollection
from utils.datacleaner import DataCleaner

    
kdc = KaggleDataCollection()
try:
    kdc.get_kaggle_dataset("lukebarousse/data-analyst-job-postings-google-search", "gsearch_jobs.csv")
except Exception as e:
    print("")

data_path = kdc.get_data_filepath()

raw_data = pd.read_csv(data_path, index_col=0)

dc = DataCleaner(raw_data)

data = (
    dc.remove_columns()
    .remove_duplicates()
    .clean_location()
    .clean_via()
    .clean_schedule_type()
    .clean_work_from_home()
    .clean_datetime('date_time')
    .clean_salary(extract_salary_from="description")
    .
    .get_cleaned_data()
)
print(data.shape)





    
    







# kdc.get_kaggle_dataset("lukebarousse/data-analyst-job-postings-google-search", "gsearch_jobs.csv")

# class DataPreprocessing():
#     pass


# if __name__ == "__main__":
#     pass
