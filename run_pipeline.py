from pathlib import Path
import pandas as pd
from src._struct.pipeline import DataPipeline

pipeline = DataPipeline(
        kaggle_dataset="lukebarousse/data-analyst-job-postings-google-search",
        file_to_download="gsearch_jobs.csv"
    )
pipeline.run_pipeline()
file = pipeline.cleaned_filepath
data = pd.read_parquet(file)
# Saving data into sub frames for working streamlit efficiently
save_to = Path('./data-subframes')
save_to.mkdir(parents=True, exist_ok=True)
salary = data[['salary_pay', 'salary_min', 'salary_max', 'salary_average']]
tokens = data[['description_tokens', 'technical_tokens', 'softskills_tokens', 'education_tokens']]
cols_to_drop = list(salary.columns) + list(tokens.columns)
jobs = data.drop(columns=cols_to_drop)
salary.to_parquet(save_to/'salary.parquet', index=False)
tokens.to_parquet(save_to/'tokens.parquet', index=False)
jobs.to_parquet(save_to/'jobs.parquet', index=False)
print(salary.columns)
print(tokens.columns)
print(jobs.columns)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    