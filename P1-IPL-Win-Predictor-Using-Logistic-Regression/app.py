import streamlit as st
import pandas as pd
import pickle

pipe = pickle.load(open('pipe1.pkl','rb'))
st.title('IPL Win Predictor')

teams = sorted(['Sunrisers Hyderabad',
 'Mumbai Indians',
 'Royal Challengers Bangalore',
 'Kolkata Knight Riders',
 'Kings XI Punjab',
 'Chennai Super Kings',
 'Rajasthan Royals',
 'Delhi Capitals',
 'Lucknow Super Giants',
 'Gujarat Titans'
 ])

col1,col2 = st.columns(2)

with col1:
    batting_team =st.selectbox('Select the batting team',teams)
with col2:
    bowling_team = st.selectbox('Select the bowling team',teams)
    
cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
       'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
       'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
       'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
       'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
       'Sharjah', 'Mohali', 'Bengaluru']

selected_city = st.selectbox('Cities',sorted(cities))

target = st.number_input('Target',min_value=0)

col3,col4,col5 = st.columns(3)
with col3 :
    score =st.number_input('Current Score',min_value=0)
with col4 :
    wickets =st.number_input('Wickets Fallen',min_value=0,max_value=9)
with col5 :
    overs = st.number_input('Overs completed',min_value=0,max_value=20)
    
if st.button('Predict Probability'):
    runs_left = target-score
    balls_left = 120 - overs*6
    wickets_left = 10-wickets
    crr = score/overs
    rrr = runs_left*6/balls_left
    df =pd.DataFrame({'BattingTeam':[batting_team],'BowlingTeam':[bowling_team],'City':[selected_city],'runs_left':[runs_left],'balls_left':[balls_left],'wickets_left':[wickets_left],'target':[target],'crr':[crr],'rrr':[rrr]})    
    result = pipe.predict_proba(df)
    r_1 = round(result[0][0]*100)
    r_2 = round(result[0][1]*100)
    st.header('Wining Probabilty ')
    st.header(f"{batting_team}  : {r_2} %")
    st.header(f"{bowling_team}  : {r_1} %")
