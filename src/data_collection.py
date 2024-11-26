import tomllib
import kaggle
from kaggle import KaggleApi

class KaggleDataCollection():
    
    def __init__(self, username: str, key: str):
        self.__username = username
        self.__key = key
        
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
    
    def get_kaggle_dataset(self, dataset: str, filename: str, path: str = "dataset/", force: bool = False) -> None:
        api = self.__kaggle_auth0()
        api.dataset_download_file(
            dataset=dataset, 
            file_name=filename,
            path=path,
            force=force
            )
        return None

if __name__ == "__main__":
    
    with open("configs/cfg.toml", "rb") as cfg:
        configs = tomllib.load(cfg)
    
    KAGGLE_USERNAME = configs["kaggle"]["username"]
    KAGGLE_APIKEY = configs["kaggle"]["key"]  
    
    kdc = KaggleDataCollection(KAGGLE_USERNAME, KAGGLE_APIKEY)
    
    kdc.get_kaggle_dataset("lukebarousse/data-analyst-job-postings-google-search", "gsearch_jobs.csv")