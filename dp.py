import streamlit as st
import os
import pickle
import numpy as np
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_option_menu import option_menu
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        width: 300px !important;  /* Adjust Sidebar Width */
        min-width: 250px !important;
    }
    /* Justify main content */
    .stMarkdown p {
        text-align: justify !important;
    }
    .Analyze {
        color: black;
    }
    </style>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    selected = option_menu("Youtube Subscribers", ["Home", "Project Overview", "subscriber_count Estimation"], 
        icons=['house', 'info-square-fill','calculator-fill'], 
        menu_icon="cast", default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "black"},
            "icon": {"color": "white", "font-size": "18px"}, 
            "nav-link": {"font-size": "18px", "text-align": "left", "margin":"0px", "--hover-color": "black"},
            "nav-link-selected": {"background-color": "blue"},
        }
    )

df=pd.read_csv("channel_details.csv")
inp_data=pd.read_csv('channel_details_final.csv')
if selected == 'Home':
    st.subheader(":green[Youtube Subscrbers_count]",divider='blue')
    st.write("YouTube subscriber count represents the number of subscribers gained per unit of effort. Creators measure their success by tracking subscriber growth per video, per month, or through specific marketing campaigns. A high-growth channel consistently attracts large numbers of new subscribers, while a slow-growing one may struggle despite regular content uploads. Optimizing content strategy, engagement, and audience targeting can significantly improve subscriber growth over time.")
    colx, coly = st.columns([0.5,0.5])
    with colx:
        st.write("Creators often track the number of subscribers gained from a sample set of videos to estimate overall channel growth. The subscriber increase from those videos is measured, and the total subscriber growth for the channel is extrapolated based on engagement patterns and audience reach. This helps creators refine their content strategies and predict future audience expansion")
        st.write("YouTube subscriber growth can also refer to the actual audience expansion from existing viewers. ")
    with coly:
        st.image(r"C:\Users\gurra\free-youtube-logo-icon-2431-thumb.png")
    st.subheader(":red[Factors affecting subscriber_count:]", divider='blue')
    colx, coly, colz = st.columns([0.2,0.6,0.2])
    st.write(":blue[View_count :] youtube Subscriber_count can estimate through the No of Views of the channel.")
    st.write(":blue[No of years:] The time period of the channel also effects the subscriber_count") 
    st.write(":blue[No_of_videos:] The No of videos also effect the subscriber_growth. ")

elif selected == 'Project Overview':
    st.write("Collected dataset of Youtube channel details through youtube API for building a predictive model where by giving channel details we can estimate Subscriber_count value.")
    st.write(":blue[Dataset Particulars:]")
    st.markdown("<hr style='margin-top: 1px;'>", unsafe_allow_html=True)
    st.write("Sample of Data:")
    st.dataframe(df.head())
    st.write("1530 channels info having 180 factors from 2012 to 2024")
    colx, coly, colz = st.columns([0.2,0.4,0.2])
    with coly:
        st.image(r"C:\Users\gurra\info.png")
    st.subheader(":green[Predictive Modeling @ Machine Learning]", divider='blue')
    st.write("For the above data, taken subscriber_count column as output (y).")
    colx, coly, colz = st.columns([0.2,0.1,0.2])
    with coly:
        st.table(df['subscriber_count'].head())
    
    st.write("Among the other columns we have taken only the below columns which are suitable as input for modeling (x).")
    st.dataframe(inp_data[['view_count', 'video_count', 'no_of_years','avg_views_per_video']].head())
    st.write(":red[Trained Multiple Machine Learning Regression Models to learn relation between X & y]")
    colx, coly, colz = st.columns([0.2,0.2,0.2])
    with coly:
        st.write("Linear Regression")
        st.write("Lasso Regression")
        st.write("Knn Regression")
        st.write("Decision Tree Regression")
        st.write("Xgboost Regressor")
        st.write("etc...")
    st.markdown("<hr style='margin-top: 1px;'>", unsafe_allow_html=True)
    st.write(":blue[Models Performance Scores:]")
    colx, coly, colz = st.columns([0.2,0.4,0.2])
    with coly:
        st.image(r"C:\Users\gurra\model_performance.png")
    st.write(":blue[Among the above models we got better performance for Lasso and Linear Regressor Regressor]")
    st.write("Lasso Trained Model is Connected for Subscriber Estimations..Click on it for Prediction")

else:
    selected_date = st.date_input("Select channel created date Date", datetime.date.today(),min_value=datetime.date(1999, 1, 1))
    selected_date= pd.to_datetime(str(selected_date)) 
    year = selected_date.year
    month = selected_date.month
    day_of_week = selected_date.weekday()

    # Numerical features
    view_count = st.number_input("Enter View Count", value=0)
    video_count = st.number_input("Enter Video Count", value=0)
    dat=pd.to_datetime('2025-05-05')
    no_of_years=(dat-selected_date).days/365 
    avg_vies=round(view_count/video_count) if video_count>0 else 1
    st.write(f'No_of_years : {no_of_years}')

    avg_views_per_video=avg_vies
    # Compile all inputs
    input_values = [view_count, video_count, no_of_years,avg_vies]
    input_array = np.array(input_values).reshape(1, -1)

    with open("scaler11.pkl", "rb") as f:
        scaler = pickle.load(f)
    input_array = scaler.transform(input_array) 
    X_test = pd.DataFrame(input_array, columns=['view_count', 'video_count', 'no_of_years',
        'avg_views_per_video'])
    # Predict button
    if st.button("Predict"):
        with open('lasso_2.pkl', "rb") as f:
            model=pickle.load(f)
            ans=model.predict(X_test)
        results_df = pd.DataFrame(data=[[view_count, video_count, no_of_years,avg_vies,ans]], columns=['view_count', 'video_count', 'no_of_years','avg_views_per_video', "Prediction"])
        st.dataframe(results_df)
