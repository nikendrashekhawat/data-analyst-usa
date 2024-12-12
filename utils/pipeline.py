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
        self.dataframe = None



    def collect_data(self) -> pd.DataFrame:
        """
        Downloads the dataset from Kaggle and loads it into a DataFrame.

        Returns
        -------
        None
        
        Raises
        ------
        FileNotFoundError
            If the dataset download fails.
        """
        print('*'*30)
        print('Collecting Data Process begins..')
        print('*'*30)

        collector = KaggleDataCollection()
        collector.get_kaggle_dataset(dataset=self.dataset, filename=self.filename, dest_path=self.dest_path)
        raw_file = collector.get_data_filepath()
        if raw_file is None:
            raise FileNotFoundError("File does not exist. Might failed to download the dataset from Kaggle.")
        self.df = pd.read_csv(raw_file, index_col=0)

        print('*'*30)
        print('Collecting Data Process finished..')
        print(f'Raw Data is downloaded and saved in `{raw_file}`')
        print('*'*30)
        return None
    

    def clean_data(self) -> pd.DataFrame:
        """
        Cleans the loaded dataset and saves it to a CSV file.

        Returns
        -------
        None
        """
        if self.df is None:
            raise ValueError("No dataset found. Please run `collect_data()` first.")
        
        print('*'*30)
        print('Cleaning Process begins..')
        print('*'*30)

        cleaner = DataCleaner(self.df)
        (
            cleaner.remove_columns()
            .remove_duplicates()
            .filter_data_analyst_jobs()
            .clean_datetime("date_time")
            .clean_location(fillna_val="Others")
            .clean_via()
            .clean_schedule_type()
            .clean_work_from_home()
            .clean_salary(extract_salary_from="description")
            .remove_punctuations(from_col="description")
            .tokenize_column("description", after_mutate='remain')
            .split_tokens(col_to_split="description_tokens", new_cols_name=['technical_tokens', 'softskills_tokens', 'education_tokens'])
            .remove_columns(['description', 'extensions'])
        )
        # Save the cleaned dataset
        cleaner.save_dataset(path_to_save=self.cleaned_filepath)
        if os.path.isfile(cleaner.cleaned_file_path):
            self.cleaned_filepath = cleaner.cleaned_file_path

        print('*'*30)
        print('Cleaning Process finished.')
        print(f'Cleaned data is saved in `{cleaner.cleaned_file_path}`')
        print('*'*30)
        return None




    def run_pipeline(self):
        """
        Runs the entire pipeline from data collection to cleaning.

        Returns
        -------
        None
        """
        self.collect_data()
        self.clean_data()
        return None


if __name__ == "__main__":
    pass

