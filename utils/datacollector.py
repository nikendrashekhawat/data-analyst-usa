from pathlib import Path
from typing import Optional
from kaggle.api.kaggle_api_extended import KaggleApi

class KaggleDataCollection():
    """
    A class to manage dataset downloads from Kaggle using the Kaggle API.

    This class handles downloading specific files from Kaggle datasets and stores
    the file path for further use.

    Attributes:
    -----------    
    _api : KaggleApi
        An instance of the Kaggle API client.

    _datapath: Path, or None
        The file path of the last downloaded dataset file.
    """
    def __init__(self):
        self._api = KaggleApi()
        self._api.authenticate()
        self._datapath = None
    
      
    def get_kaggle_dataset(self, dataset: str, filename: str, dest_path: str = None, force: bool = False) -> None:
        """
        Downloads a specific file from a Kaggle dataset.

        If the destination path is not provided, the file is saved in a default 
        directory named 'dataset' in the current working directory.

        Args:
        ----
        dataset : str
            The Kaggle dataset in the format 'owner/dataset-name'.
            
        filename : str
            The name of the file to download from the dataset.
        
        dest_path : str, optional 
            The destination directory path where the file should be saved. 
            Defaults to None (creates 'dataset' folder).
        
        force : bool, optional 
            Whether to force file download even if it exists.Defaults to False.

        Returns:
        -------
            None
        """
        dest_dir = Path(dest_path or "./dataset")
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            self._api.dataset_download_file(
            dataset=dataset, 
            file_name=filename,
            path=str(dest_dir),
            force=force
            )
        except Exception as e:
            print(f"Error downloading file: {e}")
            
        self._datapath = dest_dir / filename
        return None

        
    def get_data_filepath(self) -> Optional[Path]:
        """
        Retrieves the file path of the last downloaded dataset file.

        Returns:
        Path | None: 
            The file path of the last downloaded file if available, otherwise None.
        """
        return self._datapath


if __name__ == "__main__":
    pass