import streamlit as st
import pandas as pd
import plotly.express as px
from strm.st_extends import count_tokens
from strm.st_extends import plot_horizontal
st.set_page_config(page_title="Data Analyst Jobs (USA)", layout="wide")

st.title("Data Analyst Jobs - USA")

salary = pd.read_parquet("data-subframes/salary.parquet")
skills = pd.read_parquet("data-subframes/tokens.parquet")
jobs = pd.read_parquet("data-subframes/tokens.parquet")

median_min = salary["salary_min"].median()
median_max = salary["salary_max"].median()
average_salary = salary["salary_avg"].mean().round(2)

cont_met = st.container(border=True)
with cont_met:
    cont_met.markdown("<h2 style='text-align: center;'>Median Salary</h2>", unsafe_allow_html=True)
    met_col1, met_col2, met_col3 = st.columns(3)
    with met_col1.container():
        st.metric('Minimum Salary', median_min)  
    with met_col2.container():
        st.metric('Maximum Salary', median_max)
    with met_col3.container():
        st.metric('Average Salary', average_salary)
    
technical_skills = count_tokens(skills["technical_tokens"]).sort_values('count', ascending=False).head(10)
soft_skills = count_tokens(skills["softskills_tokens"]).sort_values('count', ascending=False).head(5)
education = count_tokens(skills["education_tokens"]).sort_values('count', ascending=False)

fig1 = plot_horizontal(
    technical_skills['tokens'], 
    technical_skills['percentage'],
    xlabel='Percentage')
st.pyplot(fig1)

st.markdown("<br><br>", unsafe_allow_html=True)
            
col_fig2, col_fig3 = st.columns(2)
with col_fig2:
    fig2 = plot_horizontal(
        soft_skills['tokens'], 
        soft_skills['percentage'],
        xlabel='Percentage')
    st.pyplot(fig2)
with col_fig3:
    fig3 = plot_horizontal(
        education['tokens'], 
        education['percentage'],
        xlabel='Percentage')
    st.pyplot(fig3)