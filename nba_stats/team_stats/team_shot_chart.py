import requests
import bs4 as bs
import pandas as pd
import json
from pprint import pprint
import numpy as np
import matplotlib.image as mpimg
from datetime import datetime
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib._png import read_png
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import re
%matplotlib inline
import matplotlib.pyplot as plt
from matplotlib.image import BboxImage
from matplotlib.transforms import Bbox, TransformedBbox
import seaborn as sns


headers = {
    }
team_id_dict = {
    'Atlanta Hawks' : [1610612737, 'ATL'],
    'Boston Celtics': [1610612738, 'BOS'], 
    'Brooklyn Nets': [1610612751, 'BKN'],
    'Charlotte Hornets': [1610612766, 'CHA'],
    'Chicago Bulls': [1610612741, 'CHI'],
    'Cleveland Cavaliers': [1610612739, 'CLE'],
    'Dallas Mavericks': [1610612742, 'DAL'],
    'Denver Nuggets': [1610612743, 'DEN'],
    'Detroit Pistons': [1610612765, 'DET'],
    'Golden State Warriors': [1610612744, 'GSW'],
    'Houston Rockets': [1610612745, 'HOU'],
    'Indiana Pacers': [1610612754, 'IND'],
    'LA Clippers':[1610612746, 'LAC'],
    'Los Angeles Lakers': [1610612747, 'LAL'],
    'Memphis Grizzlies': [1610612763, 'MEM'],
    'Miami Heat': [1610612748, 'MIA'],
    'Milwaukee Bucks': [1610612749, 'MIL'],
    'Minnesota Timberwolves': [1610612750, 'MIN'],
    'New Orleans Pelicans': [1610612740, 'NOP'],
    'New York Knicks': [1610612752, 'NYK'],
    'Oklahoma City Thunder': [1610612760, 'OKC'],
    'Orlando Magic': [1610612753, 'ORL'],
    'Philadelphia 76ers': [1610612755, 'PHI'],
    'Phoenix Suns': [1610612756, 'PHO'],
    'Portland Trail Blazers': [1610612757, 'POR'],
    'Sacramento Kings': [1610612758, 'SAC'],
    'San Antonio Spurs': [1610612759, 'SAS'],
    'Toronto Raptors': [1610612761, 'TOR'],
    'Utah Jazz': [1610612762, 'UTA'],
    'Washington Wizards': [1610612764, 'WAS'],
}



for team in team_id_dict:
    abbrev = team_id_dict[team][1]
    img_url = 'https://d2p3bygnnzw9w3.cloudfront.net/req/201901091/tlogo/bbr/{}.png'.format(abbrev)
    team_id_dict[team].append(img_url)
    
team_id_dict['Brooklyn Nets'][2] = 'https://d2p3bygnnzw9w3.cloudfront.net/req/201901171/tlogo/bbr/NJN.png'
team_id_dict['New Orleans Pelicans'][2] = 'https://d2p3bygnnzw9w3.cloudfront.net/req/201901171/tlogo/bbr/NOH.png'
team_id_dict['League Average'] = [None, 'AVG', '/home/lbianculli/desktop/nba-logo.png']



url = 'https://stats.nba.com/stats/leaguedashteamshotlocations?Conference=&DateFrom=&DateTo=&DistanceRange=By+Zone&Division=&GameScope=&GameSegment=&LastNGames=0&LeagueID=&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=&PaceAdjust=N&PerMode=Totals&Period=0&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2018-19&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&StarterBench=&TeamID=&VsConference=&VsDivision='
resp = requests.get(url, headers=headers)
resp.raise_for_status()
results = resp.json()['resultSets']


team_shot = {}
for sublist in results['rowSet']:
    team = sublist.pop(1)
    sublist = sublist[1:]
    cols = ['FGA', 'FGM', 'FG%']
    index = results['headers'][0]['columnNames']
    data = [sublist[i:i+3] for i in range(0, len(sublist), 3)]
    df = pd.DataFrame(index=index, columns=cols, data=data)
    df = df.drop('Backcourt')
    team_shot[team] = df
    
    
#multi indexing
def league_df(team_shot_dict):
    
    df_list = []
    for k, v in team_shot_dict.items():
        index = [[k]*len(v.index), v.index]
        df_list.append(pd.DataFrame(index=index, columns=v.columns, data=v.values))

    df = pd.concat(df_list)
    return df


def top_frequency(zone, mean=False, top_n=5):
    df = league_df(team_shot)
    df = df.unstack().sort_values(('FGA', zone), ascending=False)
    top_by_zone = pd.concat([df['FGA'][zone], df['FG%'][zone]], axis=1).head(top_n)
    top_by_zone.columns = ['FGA', 'FG%']
    
    if mean is True:
        means =  [df['FGA'][zone].mean(), df['FG%'][zone].mean()]
        means_df = pd.DataFrame(index=['League Average'], columns=['FGA', 'FG%'], data=[means])
        top_by_zone = top_by_zone.append(means_df)
    
    return top_by_zone #round differently?


data = top_frequency('Above the Break 3', top_n=15)
x = data['FGA']
y = data['FG%']



# NO CLIPPING
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib._png import read_png
from matplotlib.text import Annotation
from skimage.transform import resize

def annotate_with_images(data):
    sns.set_style('white')
    
    for idx, row in data.iterrows():  # for each row
        team = idx
        im = mpimg.imread(team_id_dict[team][2])  # is there a difference between read_png and plt.imread?
        im = resize(im, (125, 125), mode='reflect', anti_aliasing=True)
        xy = [row[0], row[1]] # could use loc instead

        
        imagebox = OffsetImage(im, zoom=.3)
        ab = AnnotationBbox(imagebox, xy,
                            xybox=(0., 0.),
                            xycoords='data',
                            boxcoords="offset points",
                            frameon=False,
                            bboxprops=dict(boxstyle='circle',))   
                    
            
        ax.add_artist(ab)

        
x = data['FGA']
y = data['FG%']
x_mean = x.mean().round(3)
y_mean = y.mean().round(3)


# w/o clipping
fig = plt.figure(figsize=(15,13))
fig.clf()
ax = plt.subplot(111)

annotate_with_images(data)
sns.regplot(x,y,data, ci=None)

plt.xlim([min(data['FGA'])*.9, max(data['FGA']*1.1)])
plt.ylim([min(data['FG%'])*.9, max(data['FG%']*1.1)])

t1 = plt.text(ax.get_xlim()[1]*.9, ax.get_ylim()[0]*1.03, 
              'League Average FGA: {}'.format(x_mean))  # play with coords
t2 = plt.text(ax.get_xlim()[1]*.9, ax.get_ylim()[0]*1.02,
              'League Average FG%: {}'.format(y_mean));
