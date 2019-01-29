import pandas as pd 
import numpy as np
import requests
import datetime as dt
import sklearn.preprocessing
import seaborn as sns
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import cm
import bs4 as bs
import time
import concurrent.futures
import functools
import re
import logging
from multiprocessing.dummy import Pool as ThreadPool
import concurrent.futures
from collections import defaultdict
# %store -r stats_dict
# %store -r draft_dict
# %store -r stats_df_dict

def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print('Time to complete: {:.2f}'.format(time.time() - start))
        return result
    return wrapper

def remove_comment_tags(url):
    """
    takes an input url and returns the DOM tree w/o comment tags wrapping the tables
    """
    resp = requests.get(url)

    #remove the opening comment tag
    no_open_tag = resp.text.replace("""<!--\n   <div class="table_outer_container">""",
                                    """<div class="table_outer_container">""")
    #remove closing comment tag
    no_close_tag = no_open_tag.replace("""</div>\n-->""","</div>")
    
    return no_close_tag

logger = logging.getLogger(__name__)

handler = logging.FileHandler('/home/lbianculli/Desktop/college_player_log.txt', 'w')
handler.setLevel(logging.ERROR)
file_format = logging.Formatter('%(name)s - %(levelname)s -- %(message)s') 
handler.setFormatter(file_format)

logger.addHandler(handler)

def get_player_links(year):
    '''
    Returns tuple of (player name, bref link) for input player. 
    For feeding into single_player_stats()
    '''
    
    
    college_player_links = []
    resp = requests.get('https://www.basketball-reference.com/draft/NBA_' + str(year) + '.html').text
    soup = bs.BeautifulSoup(resp, 'lxml')
    table = soup.find('table', {'id': 'stats'}).tbody
    name_link_tds = [td.a for td in table.find_all('td', {'data-stat': 'player'})]
    try:
        for td in name_link_tds:
            if td is not None:
                name_link_href = ('https://www.basketball-reference.com' + re.findall(r'(?<=").*(?=")', 
                                                                                           str(td))[0])
                name_soup = bs.BeautifulSoup(requests.get(name_link_href).text, 'lxml')
                draft_dict[year].append(name_soup.find('h1', {'itemprop': 'name'}).get_text())

        for player in draft_dict[year]:

            player_name = player
            for char in player: 
                if char in ".'":
                    player = player.replace(char, '') 
            player = player.lower().split()
            if len(player) > 2:
                player = [player[0], player[1] + player[2]]
            player = '-'.join(player)
            player_link = 'https://www.sports-reference.com/cbb/players/' + player + '-1.html'
            college_player_links.append((player_name, player_link))

        return college_player_links
    except Exception as e:
        print(year)
        
        
def single_player_stats(name_link_pair):
    '''
    Stores per-game and advanced data for input player/link within stats_dict[player_name]
    If available, stores per100 stats as well.
    '''
    
    per_poss_flag = True

    name, link = name_link_pair
    resp = remove_comment_tags(link)
    soup = bs.BeautifulSoup(resp, 'lxml')

    per_game_cols = []
    per_game_table = soup.find('table', {'id': 'players_per_game'})
    seasons = [th.get_text() for tr in per_game_table.tbody.find_all('tr') for th in tr.find('th')]
    per_game_data = [td.get_text() for tr in per_game_table.tbody.find_all('tr') for td in tr.find_all('td')]
    for th in per_game_table.thead.find_all('th')[1:]:
        if th.get_text() == '\xa0':
            per_game_cols.append('x')
        else:
            per_game_cols.append(th.get_text())

    try:
        per_poss_cols = []
        per_poss_table = soup.find('table', {'id': 'players_per_poss'})
        per_poss_data = [td.get_text() for tr in per_poss_table.tbody.find_all('tr') for td in tr.find_all('td')]
        for th in per_poss_table.thead.find_all('th')[1:]:
            if th.get_text() == '\xa0':
                per_poss_cols.append('x')
            else:
                per_poss_cols.append(th.get_text())

    except Exception as e:
        per_poss_flag = False

    advanced_cols = []
    advanced_table = soup.find('table', {'id': 'players_advanced'})
    advanced_data = [td.get_text() for tr in advanced_table.tbody.find_all('tr') for td in tr.find_all('td')]
    for th in advanced_table.thead.find_all('th')[1:]:
        if th.get_text() == '\xa0':
            advanced_cols.append('x')
        else:
            advanced_cols.append(th.get_text())

    stats_dict[name] = { 'seasons': seasons,
                        'per_game': [per_game_cols, per_game_data],
                        'advanced': [advanced_cols, advanced_data]
                       }

    if per_poss_flag is True:
        stats_dict[name]['per_poss'] = [per_poss_cols, per_poss_data]
        
        
        
        
#original would return a single list of tuples, would be best to output the same here
def multi_year_links(start, end):
    year_range = range(start, end+1)
    name_link_pairs = []
    with concurrent.futures.ThreadPoolExecutor(20) as executor:
        for year_list in executor.map(get_player_links, [year for year in year_range]):
            name_link_pairs.append(year_list) 
            
    name_link_pairs = [tuple_ for sublist in name_link_pairs for tuple_ in sublist] #flatten into single list
    return name_link_pairs

@timer
def multi_year_stats(start, end):
    name_link_pairs = multi_year_links(start, end+1)
    with concurrent.futures.ThreadPoolExecutor(20) as executor:
        executor.map(single_player_stats, [pair for pair in name_link_pairs])
        
        
        
def single_player_stats(name_link_pair):
    '''
    Stores per-game and advanced data for input player/link within stats_dict[player_name]
    If available, stores per100 stats as well.
    '''
    per_poss_flag = True

    name, link = name_link_pair
    resp = remove_comment_tags(link)
    soup = bs.BeautifulSoup(resp, 'lxml')

    per_game_cols = []
    per_game_table = soup.find('table', {'id': 'players_per_game'})
    seasons = [th.get_text() for tr in per_game_table.tbody.find_all('tr') for th in tr.find('th')]
    per_game_data = [td.get_text() for tr in per_game_table.tbody.find_all('tr') for td in tr.find_all('td')]
    for th in per_game_table.thead.find_all('th')[1:]:
        if th.get_text() == '\xa0':
            per_game_cols.append('x')
        else:
            per_game_cols.append(th.get_text())

    try:
        per_poss_cols = []
        per_poss_table = soup.find('table', {'id': 'players_per_poss'})
        per_poss_data = [td.get_text() for tr in per_poss_table.tbody.find_all('tr') for td in tr.find_all('td')]
        for th in per_poss_table.thead.find_all('th')[1:]:
            if th.get_text() == '\xa0':
                per_poss_cols.append('x')
            else:
                per_poss_cols.append(th.get_text())

    except Exception as e:
        per_poss_flag = False

    advanced_cols = []
    advanced_table = soup.find('table', {'id': 'players_advanced'})
    advanced_data = [td.get_text() for tr in advanced_table.tbody.find_all('tr') for td in tr.find_all('td')]
    for th in advanced_table.thead.find_all('th')[1:]:
        if th.get_text() == '\xa0':
            advanced_cols.append('x')
        else:
            advanced_cols.append(th.get_text())

    stats_dict[name] = { 'seasons': seasons,
                        'per_game': [per_game_cols, per_game_data],
                        'advanced': [advanced_cols, advanced_data]
                       }

    if per_poss_flag is True:
        stats_dict[name]['per_poss'] = [per_poss_cols, per_poss_data]
        
        
        
        
# logging here
no_stats = []
no_adv_stats = []
stats_df_dict = {}
for player in stats_dict:
    try:
        all_stats = stats_dict[player]
        basic_stats = all_stats['per_game']
        basic_cols = basic_stats[0]
        basic_data = basic_stats[1]
        n_cols = len(basic_cols)

        basic_data = [basic_data[i:i+n_cols] for i in range(0, len(basic_data), n_cols)]
        basic_df = pd.DataFrame(index=stats_dict[player]['seasons'], 
                                columns=basic_cols, data=basic_data)

        basic_cols_to_keep = ['PTS', 'TRB', 'AST', 'TOV']
        basic_df = basic_df[basic_cols_to_keep]

        try:
            basic_df = basic_df.astype(float)
            basic_df['AST/TO'] = (basic_df['AST'] / basic_df['TOV']).round(3)

        except ValueError as e:
            try:
                for column in basic_df:
                    basic_df[column] = basic_df[column].replace('', basic_df[column][-1])

                basic_df = basic_df[basic_cols_to_keep].astype(float)
                basic_df['AST/TO'] = (basic_df['AST'] / basic_df['TOV']).round(3)

            except ValueError:
                no_stats.append(player)    


        basic_df = basic_df.drop('TOV', axis=1)

        adv_stats = all_stats['advanced']
        adv_cols = adv_stats[0]
        adv_data = adv_stats[1]
        n_cols = len(adv_cols)

        adv_data = [adv_data[i:i+n_cols] for i in range(0, len(adv_data), n_cols)]
        adv_df = pd.DataFrame(index=stats_dict[player]['seasons'], 
                                columns=adv_cols, data=adv_data)

        # doesnt have blk% stl% etc very far back unfortunately
        try:
            adv_cols_to_keep = ['TS%', '3PAr', 'FTr', 'OWS', 'DWS', 'WS/40'] 
            adv_df = adv_df[adv_cols_to_keep]
            adv_df = adv_df.astype(float)

        except ValueError as e:
            adv_cols_to_keep = ['TS%', '3PAr', 'FTr', 'OWS', 'DWS', 'WS/40'] 
            adv_df = adv_df[adv_cols_to_keep]

            for column in adv_df:
                adv_df[column] = adv_df[column].replace('', adv_df[column][-1])

            adv_df = adv_df[adv_cols_to_keep].astype(float)

        except KeyError as e:
            no_adv_stats.append(player)
            adv_df = adv_df.replace('', np.nan)


        stats_df_dict[player] = {'per_game': basic_df, 
                                 'advanced': adv_df, 
                                }
    except Exception as e:
#         logger.error(
#             'Exception occured during handling of {} \n Description: {}\n'.format(player, e))
        logger.exception('Occured during handling of {}'.format(player))
        
