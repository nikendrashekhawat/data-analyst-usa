import streamlit as st
import pandas as pd
from strm.st_extends import count_tokens, plot_horizontal

st.set_page_config(page_title="Data Analyst Jobs (USA)", layout='wide')

salary = pd.read_parquet("data-subframes/salary.parquet")
skills = pd.read_parquet("data-subframes/tokens.parquet")
jobs = pd.read_parquet("data-subframes/tokens.parquet")
median_min = salary["salary_min"].median()
median_max = salary["salary_max"].median()
average_salary = salary["salary_avg"].mean().round(2)

st.markdown("<h1 style='text-align: center;'>Data Analyst Jobs - USA</h1>", unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)
    
technical_skills = count_tokens(skills["technical_tokens"]).sort_values('count', ascending=False).head(10)
soft_skills = count_tokens(skills["softskills_tokens"]).sort_values('count', ascending=False).head(5)
education = count_tokens(skills["education_tokens"]).sort_values('count', ascending=False)

fig1 = plot_horizontal(
    y=technical_skills['tokens'], 
    x=technical_skills['percentage'],
    xlabel='Percentage',
    cmap='cool',
    xlim=(0,80),
    bar_label=True,
    figsize=(6,3.5),
    title="Technical Skills",
    subtitle="The percentage of total jobs require these skills for a Data Analyst role."
    )

with st.container(border=True):
    st.pyplot(fig1, use_container_width=False)
    st.write(
        """
        These top 10 technical skills are more frequently sought by employers. In Data Analyst roles,
        SQL is the most in-demand skill, appearing in 56.4% of job postings. Excel and Python follow, 
        being required in 38% and 32.6% of roles, respectively, signifying their essential use in data
        cleaning, and analysis. Visualization tools like Tableau and Power BI are also 
        highly valued, appearing in over 20% of job postings, indicating a strong demand for
        presenting insights effectively. Foundational concepts like statistics, along with 
        niche skills like R, statistical analysis, and data mining, also play a significant
        role.
        """
    )
st.markdown("<br><br>", unsafe_allow_html=True)
     
fig2 = plot_horizontal(
    y=education['tokens'], 
    x=education['percentage'],
    xlabel='Percentage',
    figsize=(6,3),
    bar_label=True,
    xlim=(0, 80),
    color='steelblue',
    title='Education Requirement',
    subtitle='The percentage of total jobs require univerty/college degree'
    )
with st.container(border=True):
    st.pyplot(fig2, use_container_width=False)
    st.write(
        """
         Majority of job postings (60.7%) require candidates to hold a Bachelor's degree, 
         establishing it as the standard qualification for most roles. A Master's degree 
         is the second most sought-after qualification, required by 19.8% of jobs, 
         reflecting its importance in specialized or senior positions. 
         Certifications (5.7%) and Diplomas (3.3%) are valued for niche roles or 
         skill-based positions, while only 1.3% of job postings explicitly demand a PhD,
         highlighting its relevance in highly specialized fields.
        """)
