import os
from pathlib import Path
import kaggle
from kaggle import KaggleApi

class KaggleDataCollection():
    
    def __init__(self, username: str, key: str):
        self.__username = username
        self.__key = key
        self.__datapath = None
        
    @property    
    def username(self):
        return self.__username
    
    @username.setter
    def username(self, new_username: str):
        self.__username = new_username
    
    @property    
    def key(self):
        return self.__key
    
    @key.setter
    def key(self, new_key: str):
        self.__key = new_key
    
    
    def __kaggle_auth0(self) -> KaggleApi:
        api = kaggle.api
        api.CONFIG_NAME_USER = self.__username
        api.CONFIG_NAME_KEY = self.__key
        return api
    
    def get_kaggle_dataset(self, dataset: str, filename: str, dest_path: str = None, force: bool = False) -> None:
        
        if dest_path is None:
            if not os.path.isdir("./dataset"):
                os.mkdir("./dataset")
            dest_path = "dataset/"
                
        api = self.__kaggle_auth0()
        api.dataset_download_file(
            dataset=dataset, 
            file_name=filename,
            path=dest_path,
            force=force
            )
        self.__datapath = Path(".").resolve()/ Path("dataset") / filename
        return None

        
    def data_downloaded_filepath(self) -> Path | None:
        return self.__datapath

if __name__ == "__main__":
    pass
    
    # with open("configs/cfg.toml", "rb") as cfg:
    #     configs = tomllib.load(cfg)
    
    # KAGGLE_USERNAME = configs["kaggle"]["username"]
    # KAGGLE_APIKEY = configs["kaggle"]["key"]  
    
    # kdc = KaggleDataCollection(KAGGLE_USERNAME, KAGGLE_APIKEY)
    
    # kdc.get_kaggle_dataset("lukebarousse/data-analyst-job-postings-google-search", "gsearch_jobs.csv")