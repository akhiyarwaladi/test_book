#!/usr/bin/env python
# coding: utf-8

# ### location entropy spatio temporal

# This dataset comprises a set of users and their visits to various attractions in five theme parks (Disneyland, Epcot, California Adventure, Disney Hollywood and Magic Kindgom). The user-attraction visits are determined based on geo-tagged Flickr photos that are posted from Aug 2007 to Aug 2017 and retrieved using the Flickr API.
# 
# All user-attraction visits in each themepark are stored in a single csv file that contains the following columns/fields:
# > 
# * photoID: identifier of the photo based on Flickr.
# 
# * userID: identifier of the user based on Flickr.
# 
# * dateTaken: the date/time that the photo was taken (unix timestamp format).
# 
# * poiID: identifier of the attraction (Flickr photos are mapped to attraction based on their lat/long).
# 
# * poiTheme: category of the attraction (e.g., Roller Coaster, Family, Water, etc).
# 
# * poiFreq: number of times this attraction has been visited.
# 
# * rideDuration: the normal ride duration of this attraction.
# 
# * seqID: travel sequence no. (consecutive attraction visits by the same user that differ by <8hrs are grouped as one travel sequence).

# In[1]:


import pandas as pd 
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.options.mode.chained_assignment = None

import numpy as np
import math
import os


import plotly.express as px
import plotly.graph_objs as go
from ipywidgets import HBox, Layout, VBox
import ipywidgets

from tqdm import tqdm
from datetime import datetime

li_data = []
p_path = '/home/server/gli-data-science/akhiyar/location_entropy/data-sigir17/userVisits-sigir17/'
poi_path = '/home/server/gli-data-science/akhiyar/location_entropy/data-sigir17/poiList-sigir17/'
## all file in user visit
display(pd.DataFrame({'filename':os.listdir(p_path)}).T)
## load and combine data that we want
for idx, s_path in enumerate(os.listdir(p_path)):
    
    if idx in (0,1,3):
        city = s_path.split('.')[0].split('-')[1]
        

        # read and formatting
        data = pd.read_csv(p_path+'/'+s_path, sep=';')
        display(data.head())
        data['dateTakenRead'] = pd.to_datetime(data['takenUnix'],unit='s')
        data['city'] = city
        
        #data look like this
        print(city, data['dateTakenRead'].max(), data['dateTakenRead'].min())
        print(data[data['dateTakenRead'] >= '2012-01-01'].shape[0], data.shape[0], data['poiID'].nunique())
        

        li_data.append(data)
    
df_data = pd.concat(li_data)
print('combine data')
df_data.head()


# ---

# In[ ]:





# We create a function to calculate the Shannon Entropy according to the formula. (https://en.wiktionary.org/wiki/Shannon_entropy or refer test docs).

# * Captures: both frequency and diversity.
# * Intuitively: expected number of users visiting a location.
# 
# ```{image} ./figure/location_entropy.png
# :alt: fishy
# :class: bg-primary mb-1
# :width: 500px
# :align: center
# ```

# In[2]:


def calculateEntropy(pId, poiID, userId, poiFreq):
    l = np.sum(poiFreq[poiID==pId])
    entropy = 0
    for uId in userId[poiID==pId]:
        c = np.sum(poiFreq[np.logical_and([poiID==pId], [userId==uId])[0]])
        p = c/l
        entropy = entropy+p*math.log2(p)
    entropy = entropy*(-1)
    return entropy


# In[3]:


## define daterange we want to observe
li_range = {'all':(datetime(2007,1,1),datetime(2017,1,1)),
             'lower':(datetime(2007,1,1),datetime(2012,1,1)),
              'upper':(datetime(2012,1,1),datetime(2017,1,1))}


# ---

# In[4]:


## without multiprocessing

# li_df_le = []
# start = datetime.now()
# print("start at: {}".format(start))

# for city in df_data['city'].unique():
#     for lr in li_range:

#         data = df_data[(df_data['city'] == city) \
#               & (df_data['dateTakenRead'] >= li_range[lr][0]) & (df_data['dateTakenRead'] < li_range[lr][1])]
#         print('{}\t{} - {}'.format(city,str(li_range[lr][0].date()), str(li_range[lr][1].date())))
#         userId = data['nsid'].values
#         poiID = data['poiID'].values
#         poiFreq = data['poiFreq'].values


#         ## Now calculate and store Location Entropy of each locations and user provided in the loaded dataset
#         # print("location_id", "Entropy")
#         location_id = []
#         location_entropy = []

#         for pid in (np.unique(poiID)):
#             le = calculateEntropy(pid, poiID, userId, poiFreq)
#             location_id.append(pid)
#             location_entropy.append(le)
#             #print(pid,".",le)

#         df_le = pd.DataFrame({'location_id':location_id, 'location_entropy':location_entropy, 
#                               'city':city, 'date_split':lr})
#         li_df_le.append(df_le)
    
# df_le = pd.concat(li_df_le)   

# end = datetime.now()
# print("elapsed: {}".format(end-start))


# In[5]:


# start = datetime.now()
# print("start at: {}".format(start))

    
# def myfunc(tup_func):
#     city = tup_func[0]
#     lr = tup_func[1]
#     data = df_data[(df_data['city'] == city) \
#           & (df_data['dateTakenRead'] >= li_range[lr][0]) & (df_data['dateTakenRead'] < li_range[lr][1])]
#     print('{}\t{} - {}\n'.format(city,str(li_range[lr][0].date()), str(li_range[lr][1].date())))
#     userId = data['nsid'].values
#     poiID = data['poiID'].values
#     poiFreq = data['poiFreq'].values


#     ## Now calculate and store Location Entropy of each locations and user provided in the loaded dataset
#     location_id = []
#     location_entropy = []

#     for pid in (np.unique(poiID)):
#         le = calculateEntropy(pid, poiID, userId, poiFreq)
#         location_id.append(pid)
#         location_entropy.append(le)
        

#     df_le = pd.DataFrame({'location_id':location_id, 'location_entropy':location_entropy, 
#                           'city':city, 'date_split':lr})
#     return df_le
    

# from itertools import product
# from concurrent.futures import ProcessPoolExecutor
# # create multiprocessing job to loop each theme park and date range specified
# li_df_le = []
# with ProcessPoolExecutor(max_workers=16) as executor:
#     for r in executor.map(myfunc, list(product(df_data['city'].unique(), li_range))):
#         li_df_le.append(r)
    
# df_le = pd.concat(li_df_le)   

# end = datetime.now()
# print("end at: {}".format(end))
# print("elapsed: {}".format(end-start))


# In[6]:


# df_le_sel = df_le[df_le['city'] == 'disHolly']
# df_le_sel = df_le_sel.set_index(["location_id", "date_split"])['location_entropy'].unstack(level=1).reset_index()

# df_poi = pd.read_csv(poi_path+'POI-{}.csv'.format(city), sep=';')
# df_poi = df_poi[['poiID','poiName','theme','rideDuration']]

# df_le_sel = pd.merge(df_le_sel, df_poi, left_on='location_id', right_on='poiID', how='left')\
#             .sort_values(by='all', ascending=False)


# ---

# In[7]:


# li_dbox = []
# for city in df_data['city'].unique():
#     df_le_sel = df_le[df_le['city'] == city]
#     mean_order = int(df_le_sel['location_entropy'].mean())
    
#     df_le_sel = df_le_sel.set_index(["location_id", "date_split"])['location_entropy']\
#                             .unstack(level=1).reset_index()
    
#     df_poi = pd.read_csv(poi_path+'POI-{}.csv'.format(city), sep=';')
#     df_poi = df_poi[['poiID','poiName','theme','rideDuration']]

#     df_le_sel = pd.merge(df_le_sel, df_poi, left_on='location_id', right_on='poiID', how='left')

#     ## for display top poi by location entropy value
#     df_le_sel_dis = df_le_sel.sort_values(by='all', ascending=False)\
#                             .head(8).reset_index(drop=True)[['poiName', 'theme', 'all']]
#     df_le_sel_dis.index = np.arange(1, len(df_le_sel_dis)+1)
#     ##
    
#     fig = go.Figure()
#     for lr in li_range:
#         fig.add_trace(go.Scatter(
#             x=df_le_sel['location_id'],
#             y=df_le_sel[lr],
#             name=lr,
#             text=df_le_sel['poiName']

#         ))

    
#     fig.add_hline(y=mean_order, line_dash="dot",
#                   annotation_text="avg<br>{}".format(mean_order), 
#                   annotation_position="top right")


#     fig.update_traces(
#         hovertemplate='%{text}<br>%{y}')
    
    
#     legend_dict = \
#         legend=dict(
#                 orientation="h",
#                 yanchor="bottom",
#                 y=0.95,
#                 xanchor="left",
#                 x=0,
#                 traceorder="normal",
#                 title='',
#                 title_font_family="Courier",
#                 font=dict(
#                     family="Courier",
#                     size=16,
#                     color="black"
#                 ),
#                 bgcolor="#dfe4ea",
#                 bordercolor="Black",
#                 borderwidth=1
#             )

#     fig.update_layout( 
#                 xaxis={'showline': True, 'visible': True, 'showticklabels': True, \
#                        'showgrid': True, 'automargin': True, 'title':'id'},
#                 yaxis={'showline': False, 'visible': True, 'showticklabels': True,\
#                        'showgrid': True,  'automargin': True, 'title':'entropy'},
#                       uniformtext_minsize=8, uniformtext_mode='hide', margin=\
#                       {'l':70, 'r':70, 't':70, 'b':70},legend=legend_dict,title=city,\
#                       template='presentation', hoverlabel=dict(font=dict(family='sans-serif', size=17)))

#     f1 = go.FigureWidget(fig, layout=Layout(width='70%'))
    
#     dbox = HBox([f1, ipywidgets.HTML(df_le_sel_dis.to_html(), layout=Layout(width='30%'))])
#     li_dbox.append(dbox)


# In[8]:


# for b in li_dbox:
#     display(b,layout=Layout(width='100%',display='inline-flex',flex_flow='row wrap'))


# >
# * theme parks with family topics are always included in the top 8 largest location entropy, showing family topics are more popular for visitors.
# * we see that there is a slight entropy relationship with the period of data taken from the early period 2007 - 2012 data tends to have a small entropy which marks a theme park that is still unknown to many people, and tends to increase in 2012 and above.

# In[ ]:





# In[ ]:





# In[ ]:




