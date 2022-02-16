# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 20:07:05 2022

@author: emanu
"""
import matplotlib.pyplot as plt
import pandas as pd

arima_data = pd.read_excel("../data/Real_time_forecast_dataset_04_04_20.xlsx")

#Fix range
length=arima_data.count()
for k,i in enumerate(length):
    arima_data[arima_data.columns[k]]=arima_data[arima_data.columns[k]].shift(periods=len(arima_data)-i)
arima_data['date'] = pd.date_range(end='4/4/2020', periods=len(arima_data), freq='D')

arima_data = arima_data.set_index(arima_data['date']).drop(columns='date')

actual = pd.read_csv("../data/Realtime_cases_plot.csv")
actual.loc[-1] = ['2020-01-20T00:00:00Z',0, 0, 0, 0, 0] # adding a row
actual.index = actual.index + 1 # shifting index
actual.sort_index(inplace=True)

actual['date'] = pd.date_range(end='4/4/2020', periods=len(actual), freq='D')
actual = actual.set_index(actual['date']).drop(columns='date')

S_Korea = pd.merge(actual["South Korea"], arima_data["S. Korea"], 'left', on='date')
France = pd.merge(actual["France"], arima_data["France"], 'left', on='date')
Canada = pd.merge(actual["Canada"], arima_data["Canada"], 'left', on='date')
India = pd.merge(actual["India"], arima_data["India"], 'left', on='date')
UK = pd.merge(actual["United Kingdom"], arima_data["UK"], 'left', on='date')
#Change color
import matplotlib as mpl
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["#00a896", "#ef476f"]) 

import matplotlib.gridspec as gridspec
fig = plt.figure(figsize=(20, 13))
fig.suptitle('Forecast of COVID-19 cases: \n Actual vs Predicted',
            fontfamily='Tahoma',
            fontsize= 27,
            fontstyle='italic',
            fontweight ='extra bold',
            fontvariant='small-caps')
gs1 = gridspec.GridSpec(2, 3, figure = fig, hspace = 0.35);

ax0 = fig.add_subplot(gs1[0,0]);
ax1 = fig.add_subplot(gs1[0,1]);
ax2 = fig.add_subplot(gs1[0,2]);
ax3 = fig.add_subplot(gs1[1,0]);
ax4 = fig.add_subplot(gs1[1,1]);

ax0.plot(France, linewidth=2.5);
ax1.plot(Canada, linewidth=2.5);
ax2.plot(UK, linewidth=2.5);
ax3.plot(India, linewidth=2.5);
ax4.plot(S_Korea, linewidth=2.5);

ax0.set_title('France', fontsize=20)
ax1.set_title('Canada', fontsize=20)
ax2.set_title('UK', fontsize=20)
ax3.set_title('India', fontsize=20)
ax4.set_title('South Korea', fontsize=20);

ax0.tick_params('x',labelrotation=35)
ax1.tick_params('x',labelrotation=35)
ax2.tick_params('x',labelrotation=35)
ax3.tick_params('x',labelrotation=35)
ax4.tick_params('x',labelrotation=35)

ax0.grid(axis='y')
ax1.grid(axis='y')
ax2.grid(axis='y')
ax3.grid(axis='y')
ax4.grid(axis='y')

fig.legend(('Actual', 'Predicted'), prop={'size': 25}, bbox_to_anchor=(0.51, -0.4, 0.36, 0.8));
plt.close()
