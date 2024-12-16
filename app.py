import streamlit as st
import pandas as pd
import plotly.express as px
from strm.st_helpers import count_tokens


st.title("Data Analyst Jobs - USA")

salary = pd.read_parquet("data-subframes/salary.parquet")
skills = pd.read_parquet("data-subframes/tokens.parquet")

median_min = salary["salary_min"].median()
median_max = salary["salary_max"].median()

met_col1, met_col2 = st.columns(2)
with met_col1:
    st.metric("Minimum Median Salary", median_min,)
with met_col2:
    st.metric("Maximum Median Salary", median_max)
    
technical_skills = count_tokens(skills["technical_tokens"]).sort_values('count', ascending=False).head(10).sort_values('percentage')
fig = px.bar(technical_skills, y="tokens", x="percentage", orientation='h')
st.plotly_chart(fig)
