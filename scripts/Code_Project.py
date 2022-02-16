# -*- coding: utf-8 -*-
"""
Created on Sun Jan  2 17:37:41 2022

@author: emanu
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

arima_data = pd.read_excel("../data/Real_time_forecast_dataset_04_04_20.xlsx")

#Fix range
length=arima_data.count()
for k,i in enumerate(length):
    arima_data[arima_data.columns[k]]=arima_data[arima_data.columns[k]].shift(periods=len(arima_data)-i)
arima_data['date'] = pd.date_range(end='4/4/2020', periods=len(arima_data), freq='D')

arima_data = arima_data.set_index(arima_data['date']).drop(columns='date')

#to reset the colors run the code in the next line
#mpl.rcParams.update(mpl.rcParamsDefault)
arima_data.plot(
    kind='line', stacked=True,
    figsize = (15,6)
).set_title('Forecast of COVID-19 cases',
            fontfamily='Tahoma',
            fontsize='x-large',
            fontstyle='italic',
            fontweight ='extra bold',
            fontvariant='small-caps');


#add row at 20/01/2020 with all zeros
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

fig.legend(('Actual', 'Predicted'), prop={'size': 25}, bbox_to_anchor=(0.51, -0.4, 0.36, 0.8))
fig

covid_data = pd.read_excel("../data/CFR_data_04_04_20.xlsx", index_col=0)

#Convert to categorical variables: 'Income class' and 'Climate zones'
covid_data['Income class'] = covid_data['Income class'].astype('category')
covid_data['Climate zones'] = covid_data['Climate zones'].astype('category')


# Download from kaggle the dataset containing the flags URL
flags = pd.read_csv('../data/countries_continents_codes_flags_url.csv', usecols=[0,2])
flags.columns = flags.columns.str.title()

#Fix the dataset names in order to merge them
flags.replace('South Korea', 'S. Korea', inplace=True)
flags.replace('United States', 'USA', inplace=True)
flags.replace('United Kingdom', 'UK', inplace=True)
flags.replace('Czech Republic', 'Czechia', inplace=True)

flags = pd.merge(covid_data, flags, 'left', on='Country').set_index('Country')
flags.columns = flags.columns.str.title()
flags = flags.sort_values(by=['Cases In Thousands'],ascending= False)


# Converting links to html tags
def path_to_image_html(path):
    return '<img src="'+ path + '" width="50" >'

cols = flags.columns.tolist()
cols = cols[-1:] + cols[:-1]
flags = flags[cols]

#Display clear readble table
flags =(np.round(flags,decimals=3).rename(columns={'% People (>65)': 'People (>65)',
                                                   'Population Density/Km2': 'Population Density',
                                                   'Cases In Thousands': 'Cases',
                                                   'Population (In Millions)':'Population',
                                                   'Image_Url': 'Flag'})
).style.format(formatter={('People (>65)'): lambda x: "{:,.1f}%".format(x),
                          ('Population Density'): lambda x: "{:,} / km^2".format(x),
                          ('Cases'): lambda x: '{:.3f} K'.format(x),
                          ('Cfr'): lambda x: '{:.3f}'.format(x),
                          ('Population'): lambda x: '{:.2f} M'.format(x),
                          ('Doctors Per 1000 People'): lambda x: '{:.2f}'.format(x),
                          ('Hospital Beds Per 1000'): lambda x: '{:.2f}'.format(x),
                          ('Flag'): lambda x: path_to_image_html(x)})



styles = [dict(selector="tr:hover",props=[("background-color", "%s" % "#ffff99")]),
          dict(selector="th", props=[("font-size", "110%"),("text-align", "center")]),
          dict(selector="caption", props=[("caption-side", "top"),("font-size", "150%"),
                                          ("text-align", "center")])]



flags = (flags.set_table_styles(styles)
         .background_gradient(cmap= sns.light_palette("red", as_cmap=True), subset=['Cases'])
         .bar(subset=['Total Deaths'], color='#FB3640')
         .applymap(lambda x: 'font-weight: bold;', subset=['Cfr'])
         .set_caption("CFR Dataset"))

#flags #style object, viewable if the code is run in Jupyter Notebook
#or run the following code and then open the iman'flags.png' saved in the working directory
#import dataframe_image as dfi
#dfi.export(flags, 'CFR_Dataset.png')

summary = covid_data.describe().transpose()
summary['variance']=np.square(summary['std'])
summary = summary.drop(columns=['count', 'std','25%', '50%', '75%'])
summary.round(2)

from sklearn.tree import DecisionTreeRegressor
from sklearn import tree

#We exclude Total Deaths and Climate Zones as in the paper
X = covid_data.drop(columns=['CFR', 'Total deaths', 'Climate zones'])
y = covid_data['CFR']

# We equal the parameters as the control parameters of the corresponfing R function 'rpart' used in the paper
model = tree.DecisionTreeRegressor(criterion= "mse", # $method='anova'
                                   min_samples_split = 5, # $minsplit = 5
                                   max_depth=30, # $maxdepth
                                   min_samples_leaf=2) #$minbucket
model.fit(X,y);

plt.figure(figsize=(100,70))
features = X.columns.str.title()
tree.plot_tree(model,fontsize=40, feature_names=features,
               filled=True, node_ids=False, rounded=True)
plt.show()

from dtreeviz.trees import dtreeviz
viz = dtreeviz(model, X, y, target_name="CFR",
               feature_names= features, title='Regression Tree', scale=0.8,
               orientation="LR", show_node_labels = False,
               colors={'title':'black','text':'#14213d','arrow':'#455e89',
                       'scatter_marker':'#a01a58','tick_label':'grey','split_line':'#CED4DA'})
viz


(pd.Series(model.feature_importances_,
           index= X.columns.str.title())
   .nsmallest(10) #To plot the 5 most important variables
   .plot(kind='barh',
         title = 'Variable Importance',
         figsize = [12,6],
         table = False,
         fontsize = 13,
         color = '#2e6f95',
         align='edge', width=0.8
         ));


world = pd.read_csv("https://raw.githubusercontent.com/dbouquin/IS_608/master/NanosatDB_munging/Countries-Continents.csv")
world.replace('US', 'USA', inplace=True)
world.replace('United Kingdom', 'UK', inplace=True)
world.replace('CZ', 'Czechia', inplace=True)
world.replace('Russian Federation', 'Russia', inplace=True)
world.replace('Korea, South', 'S. Korea', inplace=True)

final = pd.merge(covid_data, world, 'left', on='Country')
final1 = final.groupby('Continent')

Africa = final1.get_group('Africa') #3 observations
Asia = final1.get_group('Asia') #13 observations
Europe = final1.get_group('Europe') #23 observation
N_America = final1.get_group('North America') #5 observation
S_America = final1.get_group('South America') #5 observation
Oceania = final1.get_group('Oceania') #1 observation

X3 = Europe.drop(columns=['CFR', 'Continent', 'Country', 'Total deaths', 'Climate zones'])
y3 = Europe['CFR']

#min split 10%, so in this case =2
model3 = tree.DecisionTreeRegressor(criterion= "mse", min_samples_split = 2, max_depth=15, min_samples_leaf=2)
model3.fit(X3,y3)
plt.figure(figsize=(80,50))
features = X3.columns
tree.plot_tree(model3, feature_names=features,filled=True, fontsize=50)
plt.show()

viz2 = dtreeviz(model3, X3, y3,
               target_name="CFR", feature_names= features,
               title='Europe Regression Tree', fontname="Arial", title_fontsize=15,
               scale=1.3, show_node_labels = False, 
               colors={'title':'black', 'text':'#14213d', 'arrow':'#455e89',
                       'scatter_marker':'#a01a58','tick_label':'grey','split_line':'#CED4DA'})
viz2


(pd.Series(model3.feature_importances_,
           index= X3.columns.str.title())
   .nsmallest(10) #To plot the 5 most important variables
   .plot(kind='barh',
         title = 'Variable Importance',
         figsize = [12,6],
         table = False,
         fontsize = 13,
         color = '#4E8d95',
         align='edge', width=0.8));


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

RMSE = 0.013
MAE = float('nan')
r2 = 0.896
Adj_r2 = 0.769

y_pred = model.predict(X)

RMSE2 = np.round(mean_squared_error(y, y_pred, squared=False), 4) 
MAE2 = np.round(mean_absolute_error(y, y_pred),4)
r2_2 = np.round(r2_score(y, y_pred),2)
Adj_r2_2 = np.round(1 - (1-r2_score(y, y_pred)) * (len(y)-1)/(len(y)-X.shape[1]-1),2)

y_pred3 = model3.predict(X3)

RMSE3 = mean_squared_error(y3, y_pred3, squared=False)
MAE3 = mean_absolute_error(y3, y_pred3)
r2_3 = r2_score(y3, y_pred3)
Adj_r2_3 = 1 - (1-r2_score(y3, y_pred3)) * (len(y3)-1)/(len(y3)-X3.shape[1]-1)

(pd.DataFrame({'RMSE':[RMSE, RMSE2, RMSE3], 'MAE' :[MAE, MAE2, MAE3],
               'R^2': [r2, r2_2, r2_3], 'Adjusted R^2': [Adj_r2, Adj_r2_2, Adj_r2_3]}, 
              index = ['Paper Model Metrics', 'Our Model Metrics', 'EU Model Metrics'])
 .style.set_caption("Models Metrics")
 .set_table_styles([{'selector': 'caption', 'props': 'caption-side: top; font-size:1.8em;'}])
 .format(formatter={('RMSE'): lambda x: "{:,.3f}".format(x), ('MAE'): lambda x: "{:,.3f}".format(x),
                    ('R^2'): lambda x: '{:,.3f}'.format(x), ('Adjusted R^2'): lambda x: '{:,.3f}'.format(x)}))




