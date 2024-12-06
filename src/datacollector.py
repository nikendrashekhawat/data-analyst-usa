import os
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

class KaggleDataCollection():
    
    def __init__(self):
        self._api = KaggleApi()
        self._api.authenticate()
        self._datapath = None
    
      
    def get_kaggle_dataset(self, dataset: str, filename: str, dest_path: str = None, force: bool = False) -> None:
        if dest_path is None:
            if not os.path.isdir("./dataset"):
                os.mkdir("./dataset")
            dest_path = "dataset/"  
        self._api.dataset_download_file(
            dataset=dataset, 
            file_name=filename,
            path=dest_path,
            force=force
            )
        self._datapath = Path(".").resolve()/ Path(dest_path) / filename
        return None

        
    def get_data_filepath(self) -> Path | None:
        return self._datapath


if __name__ == "__main__":

    dataset = "lukebarousse/data-analyst-job-postings-google-search"
    file = "gsearch_jobs.csv"
    kdc = KaggleDataCollection()
    kdc.get_kaggle_dataset(dataset=dataset, filename=file)