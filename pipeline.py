from typing import Optional
import os
import pandas as pd
from utils.datacollector import KaggleDataCollection
from utils.datacleaner import DataCleaner

class DataPipeline():
    """
    A data pipeline that downloads and cleans a dataset.

    This class manages dataset collection from Kaggle, cleaning it, 
    and saving it for web application use.

    Attributes
    ----------
    dataset : str
        The Kaggle dataset in the format 'owner/dataset-name'.
    
    filename : str
        The file to download from the Kaggle dataset.
    
    dest_path : str
        Directory where the dataset will be saved.
    
    cleaned_file_path : str
        Path to save the cleaned dataset.
    
    Methods
    -------
    collect_data():
        Downloads the dataset from Kaggle.
    
    clean_data():
        Cleans the downloaded dataset.
    
    run_pipeline():
        Runs the full pipeline from collection to cleaning.
    """

    def __init__(self, kaggle_dataset: str, file_to_download: str, dest_path: Optional[str] = None, cleaned_file_path: Optional[str]=None):
        self.dataset = kaggle_dataset
        self.filename = file_to_download
        self.dest_path = dest_path
        self.cleaned_filepath = cleaned_file_path


    def run_pipeline(self) -> pd.DataFrame:
        """
        Runs the data pipeline: download, clean, and save.

        Returns
        -------
        pd.DataFrame
            The cleaned DataFrame ready for use.
        """
        # Step 1: Download the dataset
        collector = KaggleDataCollection()
        collector.get_kaggle_dataset(dataset=self.dataset, filename=self.filename, dest_path=self.dest_path)
        
        file_path = collector.get_data_filepath()
        if file_path is None:
            raise FileNotFoundError("File does not exist. Might failed to download the dataset from Kaggle.")
        # Step 2: Load the dataset
        raw_data = pd.read_csv(file_path, index_col=0)
        # Step 3: Clean the dataset
        cleaner = DataCleaner(raw_data)
        (
            cleaner.remove_columns()
            .remove_duplicates()
            .filter_data_analyst_jobs()
            .clean_datetime("date_time")
            .clean_location()
            .clean_via()
            .clean_schedule_type()
            .clean_work_from_home()
            .clean_salary("description")
            .remove_punctuations("description")
            .tokenize_column("description", after_mutate='remain')
            .remove_columns(['description', 'extension'])
        )
        # Step 4: Save the cleaned dataset
        cleaner.save_dataset(self.cleaned_filepath)
        if os.path.isfile(cleaner.cleaned_file_path):
            self.cleaned_filepath = cleaner.cleaned_file_path
        return None


if __name__ == "__main__":
    # Example usage
    pipeline = DataPipeline(
        kaggle_dataset="lukebarousse/data-analyst-job-postings-google-search",
        file_to_download="gsearch_jobs.csv"
    )
    pipeline.run_pipeline()
    file = pipeline.cleaned_filepath
    data = pd.read_csv(file, index_col=0)
    print(data.columns)
    print(data.head())
