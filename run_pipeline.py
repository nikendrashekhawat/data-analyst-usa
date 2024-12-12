import pandas as pd
from utils.pipeline import DataPipeline

pipeline = DataPipeline(
        kaggle_dataset="lukebarousse/data-analyst-job-postings-google-search",
        file_to_download="gsearch_jobs.csv"
    )

pipeline.run_pipeline()
file = pipeline.cleaned_filepath
data = pd.read_csv(file, index_col=0)
print(data.columns)

