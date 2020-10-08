#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 13:38:19 2020

@author: ashmitha shetty
"""

import pystan
import pandas as pd
from fbprophet import Prophet
from fbprophet.diagnostics import performance_metrics,cross_validation



confirmed_global_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
death_global_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
recovered_global_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'
population_density_url = 'https://raw.githubusercontent.com/ashshetty-prog/prophet_covid19/main/data.csv'
temp_url = 'https://raw.githubusercontent.com/ashshetty-prog/prophet_covid19/main/temp_data.csv'
global max_temp,min_temp,test,y_values,yhat_values


def cases_by_date(df, case_type):
    cases_by_date = pd.melt(df, id_vars=['Province/State','Country/Region','Lat','Long'], var_name='Date', value_name=case_type)
    return cases_by_date


confirmed_df = pd.read_csv(confirmed_global_url)
death_df = pd.read_csv(death_global_url)
recovered_df = pd.read_csv(recovered_global_url)
state_avgtemp_df = pd.read_csv(temp_url).drop(columns=['pop2020','averageTemperature'])
density_df = pd.read_csv(population_density_url, usecols = ["Country/Region","density"])

confirmed_cases_df = cases_by_date(confirmed_df,'Confirmed')
death_cases_df = cases_by_date(death_df,'Death')
recovered_cases_df = cases_by_date(recovered_df,'Recovered')
confirmed_cases_df.head()
state_avgtemp_df.head()



combined_df = confirmed_cases_df.join(death_cases_df['Death']).join(recovered_cases_df['Recovered'])
cases_per_day = combined_df.groupby(['Date','Country/Region'],as_index=False)[['Confirmed','Death', 'Recovered']].sum()
cases_reg=pd.merge(cases_per_day, state_avgtemp_df, on='Country/Region')
cases_reg=pd.merge(cases_reg, density_df, on='Country/Region')

max_temp = cases_reg['averageTemperatureF'].max()
min_temp = cases_reg['averageTemperatureF'].min()
max_density = cases_reg['density'].max()
min_density = cases_reg['density'].min()

def shutdown(ds):
    date = pd.to_datetime(ds)
    if date.month > 3:
        return 0
    else:
        return 1


def temp(temp):
    if temp > (max_temp - min_temp)/2:
        return 1
    else:
        return 0
    
def density(density):
    if density > (max_density - min_density)/2:
        return 1
    else:
        return 0


def per_country(combined_df):
    cases_per_day = combined_df.groupby(['Date','Country/Region'],as_index=False)[['Confirmed','Death', 'Recovered']].sum()
    cases_by_country=pd.merge(cases_per_day, state_avgtemp_df, on='Country/Region')
    cases_by_country=pd.merge(cases_by_country, density_df, on='Country/Region')
    max_temp = cases_by_country['averageTemperatureF'].max()
    min_temp = cases_by_country['averageTemperatureF'].min()
    max_density = cases_reg['density'].max()
    min_density = cases_reg['density'].min()
    cases_by_country.tail()
    return cases_by_country
#cases_by_country.plot(kind = 'line')


def global_cases_prediction(cases_reg,type_input):
    confirmed_df = cases_reg[['Date',type_input,'averageTemperatureF','density']]
    confirmed_df.columns = ['ds','y','temp','density']
    train=confirmed_df.sample(frac=0.8,random_state=200) #random state is a seed value
    test=confirmed_df.drop(train.index)
    p = Prophet()
    p.add_regressor('shutdown')
    p.add_regressor('temp')
    p.add_regressor('density')
    p.add_seasonality(name = "daily", period = 30.5, fourier_order = 5)
    train.loc[:,'shutdown'] = train.loc[:,'ds'].apply(shutdown)
    train.loc[:,'temp'] = train.loc[:,'temp'].apply(temp)
    train.loc[:,'density'] = train.loc[:,'density'].apply(density)
    confirmed_df.head()
    p.fit(train)
    test.loc[:,'shutdown'] = test.loc[:,'ds'].apply(shutdown)
    test.loc[:,'temp'] = test.loc[:,'temp'].apply(temp)
    test.loc[:,'density'] = test.loc[:,'density'].apply(density)
    forecast = p.predict(test)
    p.plot_components(forecast)
    forecast_df = forecast.groupby('ds')['yhat'].sum()
    print(forecast.groupby('ds')['yhat'].sum())
    test.loc[:,'ds'] = test.loc[:,'ds'].apply(lambda x: pd.to_datetime(x))
    #result_val_df = forecast_df.merge(test, on=['ds'])
    #y_values += list(result_val_df['y'].values)
    #yhat_values += list(forecast_df['yhat'].values)
    p.plot_components(forecast)
    return forecast_df


def confirmed_case_prediction_per_country(cases_by_country,country,type_input):
        y_values = []
        yhat_values = []
        country_confirmed_df = cases_by_country[(cases_by_country['Country/Region'] == country)]
        print(country_confirmed_df.info())
        confirmed_df = country_confirmed_df[['Date',type_input ,'averageTemperatureF','density']]
        print(confirmed_df.info())
        confirmed_df.columns = ['ds','y','temp','density']

        train = confirmed_df[:88]
        test = confirmed_df[88:]

        p = Prophet()
        p.add_regressor('shutdown')
        p.add_regressor('temp')
        p.add_regressor('density')
        p.add_seasonality(name = "daily", period = 30.5, fourier_order = 5)

        train.loc[:,'shutdown'] = train.loc[:,'ds'].apply(shutdown)
        train.loc[:,'temp'] = train.loc[:,'temp'].apply(temp)
        train.loc[:,'density'] = train.loc[:,'density'].apply(density)
        p.fit(train)
        test.loc[:,'shutdown'] = test.loc[:,'ds'].apply(shutdown)
        test.loc[:,'temp'] = test.loc[:,'temp'].apply(temp)
        test.loc[:,'density'] = test.loc[:,'density'].apply(density)
        forecast = p.predict(test)
        p.plot(forecast)
        forecast_df = forecast.groupby('ds')['yhat'].sum()
        test.loc[:,'ds'] = test.loc[:,'ds'].apply(lambda x: pd.to_datetime(x))
        #result_val_df = forecast_df.merge(test, on=['ds'])
        #y_values += list(result_val_df['y'].values)
        #yhat_values += list(forecast_df['yhat'].values)
        p.plot_components(forecast)
        return forecast_df
    
    
'''
from sklearn.metrics import mean_squared_error
from math import sqrt

rms = sqrt(mean_squared_error(y_values, yhat_values))
print(rms)

'''
exit = 0
while (exit == 0):
    user_input = input('Choose global or country for prediction:')
    type_input = input('Choose between Recovered, Deaths and Confirmed for prediction:')
    if user_input == 'global':
        pd = global_cases_prediction(cases_reg,type_input)
        exit = 1
    elif (user_input in combined_df['Country/Region'].unique()):
        per_country_cases = per_country(cases_reg)
        pd = confirmed_case_prediction_per_country(per_country_cases,user_input,type_input)
        exit = 1
    else:
        print('Choosen country please select from below:')
        print(combined_df['Country/Region'].unique())
        
    


df_cv = cross_validation(p,horizon = '15 days')
df_p = performance_metrics(df_cv)
print(pd.head())
df_p.to_csv(r'/Users/shetashm/Downloads/'+ user_input+'_Trial.csv', index = False, header=True)
pd.to_csv(r'/Users/shetashm/Downloads/'+ user_input+'_Prediction.csv', index = False, header=True)