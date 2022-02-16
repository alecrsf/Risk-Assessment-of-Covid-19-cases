# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 19:55:32 2022

@author: emanu
"""
import numpy as np
import pandas as pd
import seaborn as sns

covid_data = pd.read_excel("../data/CFR_data_04_04_20.xlsx", index_col=0)

#Convert to categorical variables: 'Income class' and 'Climate zones'
covid_data['Income class'] = covid_data['Income class'].astype('category')
covid_data['Climate zones'] = covid_data['Climate zones'].astype('category')

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

#Move it to another chunk not displayed
# Converting links to html tags-----------------------------------------------------------
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
                          ('Population Density'): lambda x: "{:,} /$km^2$".format(x),
                          ('Cases'): lambda x: '{:.3f}$K$'.format(x),
                          ('Cfr'): lambda x: '{:.3f}'.format(x),
                          ('Population'): lambda x: '{:.2f}$M$'.format(x),
                          ('Doctors Per 1000 People'): lambda x: '{:.2f}'.format(x),
                          ('Hospital Beds Per 1000'): lambda x: '{:.2f}'.format(x),
                          ('Flag'): lambda x: path_to_image_html(x)})

styles = [
    dict(selector="tr:hover",
         props=[("background-color", "%s" % "#ffff99")]),
    dict(selector="th", props=[("font-size", "110%"),
                               ("text-align", "center")]),
    dict(selector="caption", props=[("caption-side", "top"),
                                    ("font-size", "150%"),
                                    ("text-align", "center")])]

flags = (flags.set_table_styles(styles)
          .background_gradient(cmap= sns.light_palette("red", as_cmap=True), subset=['Cases'])
          .bar(subset=['Total Deaths'], color='#FB3640')
          .applymap(lambda x: 'font-weight: bold;', subset=['Cfr'])
          .set_caption("CFR Dataset"))
