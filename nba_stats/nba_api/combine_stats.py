import requests
import bs4 as bs
import pandas as pd
import json
import pickle
from pprint import pprint
import numpy as np
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import re
%matplotlib inline
# %store -r player_id_dict
import matplotlib.pyplot as plt
import seaborn as sns


url = 'https://stats.nba.com/stats/scoreboard/?GameDate=01/11/2019&LeagueID=00&DayOffset=0'
headers = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.110 Safari/537.36'}
    
def check_endpoints(url, print_on=True):
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    results = resp.json()['resultSets']
    
    if print_on == True:
        pprint(results[0].keys())
        if 'name' in results[0].keys():
            for i in results:
                pprint(i['name'])
            print(len(results), 'list(s)')
            print(25* '-')
        
    df_list = []
    for i in results:
        cols = i['headers']
        data = i['rowSet']
        df_list.append(pd.DataFrame(columns=cols, data=data))
        
    return results, df_list

combine_stats_results, combine_stats_dfs = check_endpoints('https://stats.nba.com/stats/draftcombinestats?LeagueID=00&SeasonYear=2018-19')
drill_results, drill_dfs = check_endpoints('https://stats.nba.com/stats/draftcombinedrillresults?LeagueID=00&SeasonYear=2018-19')

combine_stats_df = combine_stats_dfs[0]
drill_df = drill_dfs[0]
stationary_df = stationary_dfs[0]
spot_df = spot_dfs[0]

# combine_stats_df.columns
cols_to_keep = ['PLAYER_NAME', 'HEIGHT_W_SHOES_FT_IN', 'WINGSPAN_FT_IN', 'STANDING_REACH_FT_IN']
combine_stats = combine_stats_df[cols_to_keep].set_index('PLAYER_NAME')

# drill_df.columns
drill_df = drill_df[drill_df.columns[4:-1]]
drill_stats = drill_df.set_index('PLAYER_NAME')

combine_df = pd.concat((combine_stats, drill_stats), axis=1)
