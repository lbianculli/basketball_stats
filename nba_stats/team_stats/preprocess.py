import numpy as np
import pandas as pd
import pickle
from collections import defaultdict

team_map_dict = {
    'Atlanta Hawks' : 'Atlanta',
    'Boston Celtics': 'Boston', 
    'Brooklyn Nets': 'Brooklyn',
    'Charlotte Hornets': 'Charlotte',
    'Chicago Bulls': 'Chicago',
    'Cleveland Cavaliers': 'Cleveland',
    'Dallas Mavericks': 'Dallas',
    'Denver Nuggets': 'Denver',
    'Detroit Pistons': 'Detroit',
    'Golden State Warriors': 'Golden State',
    'Houston Rockets': 'Houston',
    'Indiana Pacers': 'Indiana',
    'LA Clippers':'LA Clippers',
    'Los Angeles Lakers': 'LA Lakers',
    'Memphis Grizzlies': 'Memphis',
    'Miami Heat': 'Miami',
    'Milwaukee Bucks': 'Milwaukee',
    'Minnesota Timberwolves': 'Minnesota',
    'New Orleans Pelicans': 'New Orleans',
    'New York Knicks': 'New York',
    'Oklahoma City Thunder': 'Oklahoma City',
    'Orlando Magic': 'Orlando',
    'Philadelphia 76ers': 'Philadelphia',
    'Phoenix Suns': 'Phoenix',
    'Portland Trail Blazers': 'Portland',
    'Sacramento Kings': 'Sacramento',
    'San Antonio Spurs': 'San Antonio',
    'Toronto Raptors': 'Toronto',
    'Utah Jazz': 'Utah',
    'Washington Wizards': 'Washington',
}

data_dict = {}
scores_dict = {}

data_dict['2018-19'] = data_1819
scores_dict['2018-19'] = scores_1819

# get rid of keys without scores
for year in list(scores_dict.keys()):
    for date in list(scores_dict[year].keys()):
        if scores_dict[year][date] is None:
            del scores_dict[year][date]
            
 
combined_stats = defaultdict(defaultdict)

# combine scores, basic, advanced into one dict
for year in list(scores_dict.keys()):
    for date in list(scores_dict[year].keys()):
        score_df = scores_dict[year][date]
        basic_df = data_dict[year][date][0]
        advanced_df = data_dict[year][date][1]
        defense_df = data_dict[year][date][2]
        offense_df = data_dict[year][date][3]
        combined_stats[year][date] = [score_df, basic_df, advanced_df, defense_df, offense_df]
            
            
combined_scores = defaultdict(defaultdict)

# this will return concatenated score DF
for year in list(combined_stats.keys()):  
    for date in list(combined_stats[year].keys()):
        scores = combined_stats[year][date][0]

        home_scores = scores.iloc[:, :2]
        away_scores = scores.iloc[:, 2:-1]

        away_scores.columns = ['team', 'points_for',]
        home_scores.columns = ['team', 'points_for',]

        for game in range(scores.shape[0]):
            away_scores['points_against']  = home_scores['points_for']
            home_scores['points_against'] = away_scores['points_for']


        all_scores = pd.concat([home_scores, away_scores])
        # combined_scores.index = np.arange(combined_scores.shape[0])
        all_scores.index = all_scores['team']
        all_scores = all_scores.drop('team',axis=1).astype(int)
        all_scores['won_game'] = 0

        for team in range(all_scores.shape[0]):
            if all_scores['points_for'][team] > all_scores['points_against'][team]:
                all_scores['won_game'][team] = 1
        assert all_scores['won_game'].mean() == .5

        combined_scores[year][date] = all_scores
        
        
        
final_data_dict = defaultdict(defaultdict)

for year in list(combined_stats.keys()):  # this will return concatenated score DF
    for date in list(combined_stats[year].keys()):
        basic = combined_stats[year][date][1]
        advanced = combined_stats[year][date][2]
        defense = combined_stats[year][date][3]
        offense = combined_stats[year][date][4]
        scores = combined_scores[year][date]
        teams_to_drop = []
        new_index = []

        for team in list(basic.index):
            split = team.split(' ')

            if len(split) is 3:
                if split[-1] == 'Lakers':
                    city = 'LA Lakers'
                if split[-1] == 'Blazers':
                    city = 'Portland'
                elif split[-1] != 'Lakers' and split[-1] != 'Blazers':  # really not sure why it needs to be this
                    city = split[0] + ' ' + split[1]

            elif len(split) == 2:
                if split [-1] == 'Clippers':
                    city = 'LA Clippers'
                else:
                    city = split[0]

            if city in list(scores.index):
                new_index.append(team)
            else:
                teams_to_drop.append(team)

        basic = basic.drop(teams_to_drop, axis=0)
        basic.index = new_index
        basic['SPG'] = basic['STL'] / basic['GP']
        basic['BPG'] = basic['BLK'] / basic['GP']
        basic = basic.drop(['STL', 'BLK'], axis=1)
        
        advanced = advanced.drop(teams_to_drop, axis=0).drop('GP', axis=1)
        advanced.index = new_index
        
        defense = defense.drop(teams_to_drop, axis=0).drop('GP', axis=1)
        defense.index = new_index
        
        offense = offense.drop(teams_to_drop, axis=0).drop('GP', axis=1)
        offense.index = new_index
        
        combined = pd.concat([basic, advanced, defense, offense], axis=1).drop(['TEAM_ID'], axis=1)

        final_data_dict[year][date] = combined
           

