# -*- coding: utf-8 -*-
"""
Created on Thu Sep 07 21:01:17 2020

"""
#################################################################
import streamlit as st
st.set_page_config(layout="wide")
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib
import plotly
import datetime
#matplotlib.use( 'tkagg' )
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from pandas.tseries.offsets import DateOffset
from statsmodels.tsa.arima_model import ARIMAResults
from plotly import graph_objs as go
import warnings
warnings.filterwarnings("ignore")
##################################################################
st.header('Model Deployment: CO2_Emission_Forecasting')
def plot_result_data():
        #plt.figure(figsize=(10,8))
        st.line_chart(data, y='CO2')
        #st.line_chart(model_final.fittedvalues, y='CO2')
        #st.subheader('Forecast')
        #plt.legend(loc='upper left', fontsize=8)
        st.pyplot()

def plot_forecasted_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=forecast_data.index,y=forecast_data['CO2'], name="CO2 Emission"))
        st.plotly_chart(fig,use_container_width=True)

def user_input_features():
    features = st.slider("No of Years to predict : ", min_value=1, max_value=100, value=1, step=1)
    return features

dateparse = lambda x: pd.to_datetime(x, format = '%Y')
data = pd.read_excel("CO2 dataset.xlsx", parse_dates = ['Year'], index_col ='Year', date_parser = dateparse)

st.subheader(" CO2 emission should be predicted for how many years ?")   
df = user_input_features()+1
st.info(f'{df-1} Years')

model_final = ARIMA(data['CO2'],order = (5,1,3))
model_final = model_final.fit()
model_final.fittedvalues.tail()
# load the model from disk

future_dates=[data.index[-1]+ DateOffset(years=x)for x in range(0,df)]
forecast_data=pd.DataFrame(index=future_dates[1:],columns=data.columns)
end = len(data)+len(forecast_data)
forecast_data['CO2'] = model_final.predict(start = 215, end = end , dynamic= True)
st.subheader('Forecasted Data')
plot_forecasted_data()
# Forecasted Values 
st.subheader(f'Predicted Values for next {df-1} years')
st.write(forecast_data.tail(df))
st.subheader('Predicted Result : Overview ')
plot_result_data()

#loaded_model = pickle.load(open('Forecast_model.pkl','rb'))
#prediction = loaded_model.predict(df)







