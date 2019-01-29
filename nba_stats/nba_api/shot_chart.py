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
from matplotlib import rcParams


url = 'https://stats.nba.com/stats/scoreboard/?GameDate=01/11/2019&LeagueID=00&DayOffset=0'
headers = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.110 Safari/537.36'}



### checking date ranges
start = 20170101
end = 20180101
headers = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.110 Safari/537.36'}

def get_shot_chart(player_id, season):
    
    player_id = str(player_id)
    url = 'https://stats.nba.com/stats/shotchartdetail?AheadBehind=&ClutchTime=&ContextFilter=&ContextMeasure=FGA&DateFrom=&DateTo=&EndPeriod=&EndRange=&GameID=&GameSegment=&LastNGames=0&LeagueID=00&Location=&Month=0&OpponentTeamID=0&Outcome=&Period=0&PlayerID={}&PlayerPosition=&PointDiff=&Position=&RangeType=&RookieYear=&Season={}&SeasonSegment=&SeasonType=Regular+Season&StartPeriod=&StartRange=&TeamID=0&VsConference=&VsDivision='.format(player_id, season)
    resp = requests.get(url, headers = headers)
    resp.raise_for_status()
    results = resp.json()['resultSets']
    
    cols = results[0]['headers']
    shot_data = results[0]['rowSet']
    return pd.DataFrame(columns=cols, data=shot_data)
    
    
def check_endpoints(url, print_on=False):
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    results = resp.json()['resultSets']
    
    if print_on == True:
        print(results[0].keys())
        if 'name' in results[0].keys():
            for i in results:
                print(i['name'])
            print(25* '-')
        
    df_list = []
    for i in results:
        cols = i['headers']
        data = i['rowSet']
        df_list.append(pd.DataFrame(columns=cols, data=data))
        
    return results, df_list



shot_df = get_shot_chart(2544, '2018-19')
    
    
    
player_name = shot_df['PLAYER_NAME'][0]
shot_df = shot_df[['SHOT_ZONE_AREA', 'SHOT_ZONE_RANGE', 'SHOT_DISTANCE', 'LOC_X', 'LOC_Y', 'SHOT_MADE_FLAG']]
shot_df.name = player_name


from matplotlib.patches import Circle, Rectangle, Arc

def draw_court(ax=None, color='black', lw=2, outer_lines=True):
    # If an axes object isn't provided to plot onto, just get current one
    if ax is None:
        ax = plt.gca()

    hoop = Circle((0, 0), radius=7.5, linewidth=lw, color=color, fill=False)

    backboard = Rectangle((-30, -7.5), 60, -1, linewidth=lw, color=color)
    outer_box = Rectangle((-80, -47.5), 160, 190, linewidth=lw, color=color,
                          fill=False)
    inner_box = Rectangle((-60, -47.5), 120, 190, linewidth=lw, color=color,
                          fill=False)

    top_free_throw = Arc((0, 142.5), 120, 120, theta1=0, theta2=180,
                         linewidth=lw, color=color, fill=False)
    
    bottom_free_throw = Arc((0, 142.5), 120, 120, theta1=180, theta2=0,
                            linewidth=lw, color=color, linestyle='dashed')
    restricted = Arc((0, 0), 80, 80, theta1=0, theta2=180, linewidth=lw,
                     color=color)
    corner_three_a = Rectangle((-220, -47.5), 0, 140, linewidth=lw,
                               color=color)
    corner_three_b = Rectangle((220, -47.5), 0, 140, linewidth=lw, color=color)
    three_arc = Arc((0, 0), 475, 475, theta1=22, theta2=158, linewidth=lw,
                    color=color)
    center_outer_arc = Arc((0, 422.5), 120, 120, theta1=180, theta2=0,
                           linewidth=lw, color=color)
    center_inner_arc = Arc((0, 422.5), 40, 40, theta1=180, theta2=0,
                           linewidth=lw, color=color)

    # List of the court elements to be plotted onto the axes
    court_elements = [hoop, backboard, outer_box, inner_box, top_free_throw,
                      bottom_free_throw, restricted, corner_three_a,
                      corner_three_b, three_arc, center_outer_arc,
                      center_inner_arc]

    if outer_lines:
        # Draw the half court line, baseline and side out bound lines
        outer_lines = Rectangle((-250, -47.5), 500, 470, linewidth=lw,
                                color=color, fill=False)
        court_elements.append(outer_lines)

    # Add the court elements onto the axes
    for element in court_elements:
        ax.add_patch(element)
        
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    
    return ax


def create_shot_chart(shot_data, figsize=(13,12)):
    makes = shot_data.loc[shot_data['SHOT_MADE_FLAG'] == 1]
    misses = shot_data.loc[shot_data['SHOT_MADE_FLAG']== 0]

    plt.figure(figsize=figsize)
    plt.scatter(makes.LOC_X, makes.LOC_Y, alpha=.3, color='g')
    plt.scatter(misses.LOC_X, misses.LOC_Y, alpha=.3, marker='x', color='r')
    draw_court()
    plt.title(label='{} Shot Chart \n 2014-15 Season'.format(shot_data.name), 
                  fontsize=16, pad=10)
    plt.xlim(250,-250)
    plt.ylim(422.5,-47.5)

    plt.show()
    
def joint_shot_chart(shot_data, figsize=(13,12)):
    # jointplot has 3 axes
    
    cmap = plt.cm.gist_heat_r
    joint_chart = sns.jointplot(shot_data['LOC_X'], shot_data['LOC_Y'], kind='hex', space=0, 
                                xlim=(-250, 250), ylim=(422.5, -47.5), color=cmap(.3), cmap=cmap)
    joint_chart.fig.set_size_inches(figsize)
    
    ax1 = joint_chart.ax_joint
    draw_court(ax1)
    
    joint_chart.ax_marg_x.set_axis_off()
    joint_chart.ax_marg_y.set_axis_off()
    joint_chart.ax_joint.set_axis_off()
    
    ax1.set_title(label='{} Shot Distribution \n 2014-15 Season'.format(shot_data.name), 
                  fontsize=16, pad=130)
    
    ax1.tick_params(labelbottom=False, labelleft=False)
    
    
    
def create_shot_chart(shot_data, figsize=(13,12)):
    mpl.rcParams['axes.facecolor'] = '#efe6bf'
    makes = shot_data.loc[shot_data['SHOT_MADE_FLAG'] == 1]
    misses = shot_data.loc[shot_data['SHOT_MADE_FLAG']== 0]

    plt.figure(figsize=figsize)
    plt.scatter(makes.LOC_X, makes.LOC_Y, alpha=.3, color='g')
    plt.scatter(misses.LOC_X, misses.LOC_Y, alpha=.3, marker='x', color='r')
    draw_court()
    plt.title(label='{} Shot Chart \n 2014-15 Season'.format(shot_data.name), 
                  fontsize=16, pad=10)
    plt.xlim(250,-250)
    plt.ylim(422.5,-47.5)

    plt.show()
    
    
import logging
from selenium import webdriver
import bs4 as bs
import re
from selenium.webdriver.chrome.options import Options
import pickle


player_id_dict = {}
logger = logging.getLogger(__name__)
file_handler = logging.FileHandler('player_id_errors.log')
file_handler.setLevel(logging.WARNING)
logger.addHandler(file_handler)

chrome_options = Options()
# chrome_options.add_argument('--headless')
player_list_url = 'https://stats.nba.com/players/list/?Historic=Y'
driver = webdriver.Chrome(options=chrome_options)  # do i need headers?
driver.get(player_list_url)
dom = driver.page_source
driver.implicitly_wait(10)
driver.quit()
print('Driver closed')  # setting a timeout limit?

soup = bs.BeautifulSoup(dom, 'lxml')
table = soup.find('div', {'class': 'large-10 columns'})  # or soup.find('div', {class:'stats-players-page'})
player_uls = soup.find_all('ul', {'class': 'players-list__names'})

for unordered_list in player_uls:
    for li in unordered_list.find_all('li'):  # test with just the first one
        try:
            player = li.get_text()  # last, first. Something like below
            player = ' '.join(player.split(',')[::-1]).strip()
            link = li.a.get('href')
            match = re.search(r'(?<=/)\d*(?=/)', link)  # pattern, string
            if match:
                player_id_dict[player] = int(match.group())

        except Exception as e:
            logging.exception(e, player)
            
            
            
            
def get_shot_data(player_id, season, last_n=0):
    
    if isinstance(player_id, str):
        player_id = player_id_dict[player_id]
    
    player_id = str(player_id)
    url = 'https://stats.nba.com/stats/shotchartdetail?AheadBehind=&ClutchTime=&ContextFilter=&ContextMeasure=FGA&DateFrom=&DateTo=&EndPeriod=&EndRange=&GameID=&GameSegment=&LastNGames={}&LeagueID=00&Location=&Month=0&OpponentTeamID=0&Outcome=&Period=0&PlayerID={}&PlayerPosition=&PointDiff=&Position=&RangeType=&RookieYear=&Season={}&SeasonSegment=&SeasonType=Regular+Season&StartPeriod=&StartRange=&TeamID=0&VsConference=&VsDivision='.format(last_n, player_id, season)
    resp = requests.get(url, headers = headers)
    resp.raise_for_status()
    results = resp.json()['resultSets']
    
    cols = results[0]['headers']
    shot_data = results[0]['rowSet']
    return pd.DataFrame(columns=cols, data=shot_data)

    
def summary_offense(player, season='2018-19'):
    '''
    returns some offensive stats for the player: ORtg, TS, USG, %FG unassisted
    '''
    _, adv_df_list = check_endpoints('https://stats.nba.com/stats/leaguedashplayerstats?College=&Conference=&Country=&DateFrom=&DateTo=&Division=&DraftPick=&DraftYear=&GameScope=&GameSegment=&Height=&LastNGames=0&LeagueID=&Location=&MeasureType=Advanced&Month=0&OpponentTeamID=0&Outcome=&PORound=&PaceAdjust=N&PerMode=Totals&Period=0&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season={}&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&StarterBench=&TeamID=&TwoWay=&VsConference=&VsDivision=&Weight='.format(season))
    adv_df = adv_df_list[0]

    _, scoring_df_list = check_endpoints('https://stats.nba.com/stats/leaguedashplayerstats?College=&Conference=&Country=&DateFrom=&DateTo=&Division=&DraftPick=&DraftYear=&GameScope=&GameSegment=&Height=&LastNGames=0&LeagueID=&Location=&MeasureType=Scoring&Month=0&OpponentTeamID=0&Outcome=&PORound=&PaceAdjust=N&PerMode=Totals&Period=0&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season={}&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&StarterBench=&TeamID=&TwoWay=&VsConference=&VsDivision=&Weight='.format(season))
    scoring_df = scoring_df_list[0]
    
    adv_player = adv_df.loc[adv_df.PLAYER_NAME == player]
    adv_player = adv_player[['OFF_RATING', 'TS_PCT', 'USG_PCT']]
    
    scoring_player = scoring_df.loc[scoring_df.PLAYER_NAME == player]
    pct_uast = scoring_player['PCT_UAST_FGM']
    
    adv_player['PCT_UAST'] = pct_uast.values[0]
    
    return adv_player.astype(float)


def summary_clutch(player, season):
    '''
    Returns some summary clutchtime stats, clutchtime defined as 
    last five mins of a game, +/- 5 points
    '''
    player_id = player_id_dict[player]
    _, df_list = check_endpoints('https://stats.nba.com/stats/playerdashboardbyclutch?DateFrom=&DateTo=&GameSegment=&LastNGames=0&LeagueID=&Location=&MeasureType=Advanced&Month=0&OpponentTeamID=0&Outcome=&PORound=&PaceAdjust=N&PerMode=Totals&Period=0&PlayerID={}&PlusMinus=N&Rank=N&Season={}&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&VsConference=&VsDivision='.format(player_id, season))
    five_min_five_pt = df_list[-5]
    clutch_summary = five_min_five_pt[['NET_RATING', 'TS_PCT', 'USG_PCT']]
    
    return clutch_summary.astype(float)


def create_shot_chart(player, season, last_n=0, summary_stats=True,
                      clutch_stats=True, figsize=(13,12)):   
    '''
    Creates a shot chart for player with optionality for aditional 
    summary stats and clutchtime stats
    '''
    rcParams['axes.facecolor'] = '#efe6bf'
    plt.figure(figsize=figsize)
    
    shot_data =  get_shot_data(player, season, last_n)
    shot_data.name = player
    makes = shot_data.loc[shot_data['SHOT_MADE_FLAG'] == 1]
    misses = shot_data.loc[shot_data['SHOT_MADE_FLAG']== 0]

    plt.scatter(makes.LOC_X, makes.LOC_Y, alpha=.5, color='g')
    plt.scatter(misses.LOC_X, misses.LOC_Y, alpha=.5, marker='x', color='r')
    ax = draw_court()

    
    plt.xlim(250,-250)
    plt.ylim(422.5,-47.5)
    
    if clutch_stats == True:
        clutch = summary_clutch(player, season)
        clutch_ortg = ax.text(245, 360, 'Clutch Time Net Rating: {}'.format(
            clutch['NET_RATING'].values[0]), fontsize=14)
        clutch_ts = ax.text(245, 375, 'Clutch Time True Shooting: {}%'.format(
            (clutch['TS_PCT'].values[0]*100).round(3)), fontsize=14)
        clutch_usg = ax.text(245, 390, 'Clutch Time Usage: {}%'.format(
            clutch['USG_PCT'].values[0]*100), fontsize=14)
        
    if summary_stats == True:
        summary = summary_offense(player, season)
        ortg = ax.text(-125, 360, 'Offensive Rating: {}'.format(
            summary['OFF_RATING'].values[0]), fontsize=14)

        ts = ax.text(-125, 375, 'True Shooting: {}%'.format(
            (summary['TS_PCT'].values[0]*100).round(3)), fontsize=14)

        usg = ax.text(-125, 390, 'Usage: {}%'.format(
            (summary['USG_PCT'].values[0]*100).round(3)), fontsize=14)

        pct_unassisted = ax.text(-125, 405, 'FG Unassisted: {}%'.format(
           summary['PCT_UAST'].values[0]*100), fontsize=14)
    
    
    if last_n == 0:
        plt.title(label='{} Shot Chart \n {} Season'.format(player, season), 
                      fontsize=16, pad=10)
    else:
        plt.title(label='{} Shot Chart \n Last {} games'.format(player, last_n), 
                      fontsize=16, pad=10)
    
    plt.show()
