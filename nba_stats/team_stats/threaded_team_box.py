# going from async to threading

import requests
import bs4 as bs
import pandas as pd
import json
import numpy as np
import datetime
from collections import defaultdict
import threading
from Queue import Queue

headers = {

    'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML like Gecko) Chrome/72.0.3626.81 Safari/537.36',
}

def remove_comment_tags(url, timeout=10):
    """
    takes an input url and returns the DOM tree w/o comment tags wrapping the tables
    """
    resp = requests.get(url, timeout=timeout)

    #remove the opening comment tag
    no_open_tag = resp.text.replace("""<!--\n   <div class="table_outer_container">""",
                                    """<div class="table_outer_container">""")
    #remove closing comment tag
    no_close_tag = no_open_tag.replace("""</div>\n-->""","</div>")

    return no_close_tag


def team_basic_box(url, filter_=True):
    ''' returns DF of basic box data for all teams at given date. Dont really need '''

    try:
        resp = requests.get(url, headers=headers,timeout=5)
        stats = resp.json()
        col_names = stats['resultSets'][0]['headers']
        col_names.remove('TEAM_NAME')
        data = stats['resultSets'][0]['rowSet']
        team_names = [team.pop(1) for team in data]
        df = pd.DataFrame(index=team_names, data=data, columns=col_names)

        if filter_ is True:  # use team id to map between DFs
            keep_cols = df.columns.to_series()[['TEAM_ID', 'GP', 'STL', 'BLK']]  
            df = df[keep_cols]

        return df

    except Exception as e:
        print(e)



def team_advanced_box(url, filter_=True):
    ''' returns DF of advanced box score for all teams at given date '''
    try:
        resp = requests.get(url, headers=headers,timeout=5)
        stats = resp.json()
        col_names = stats['resultSets'][0]['headers']
        col_names.remove('TEAM_NAME')
        data = stats['resultSets'][0]['rowSet']
        team_names = [team.pop(1) for team in data]
        df = pd.DataFrame(index=team_names, data=data, columns=col_names)

        if filter_ is True:
            keep_cols = df.columns.to_series()[['TEAM_ID', 'GP', 'W_PCT', 'OFF_RATING', 'PACE',
                                                'DEF_RATING', 'AST_TO', 'OREB_PCT', 'DREB_PCT', 'TS_PCT']]
            df = df[keep_cols]

        return df

    except Exception as e:
        print(e)



def gen_offense(url, filter_=True):
    ''' returns DF of general offense data for all teams at given date '''
    try:
        resp = requests.get(url, headers=headers,timeout=5).json()
        offense_results = resp['resultSets']  # where this dict will be the json
        cols = offense_results[0]['headers']
        cols.remove('TEAM_NAME')
        data = offense_results[0]['rowSet']
        team_names = [team.pop(1) for team in data]
        offense_df = pd.DataFrame(index=team_names, data=data, columns=cols)
        offense_df = offense_df.drop(['FGA_FREQUENCY', 'FGM', 'FGA', 'FG_PCT', 'EFG_PCT', 'FG2M', 'FG2A', 'FG3M', 'FG3A'], axis=1)

        return offense_df

    except Exception as e:
        print(e)


def gen_defense(two_pt_url, three_pt_url):
    ''' Scrapes two URLs for 2pt and 3pt defense, returns combined DF '''
    try:
        resp1 = requests.get(two_pt_url, headers=headers, timeout=5).json()
        two_pt_results = resp1['resultSets']  # where this dict will be the json
        two_pt_cols = two_pt_results[0]['headers']
        two_pt_cols.remove('TEAM_NAME')
        two_pt_data = two_pt_results[0]['rowSet']
        two_pt_names = [team.pop(1) for team in two_pt_data]
        two_pt_df = pd.DataFrame(index=two_pt_names, data=two_pt_data, columns=two_pt_cols)
        two_pt_df.columns = ['TEAM_ID', 'TEAM_ABBREVIATION', 'GP', 'G',
                'FREQ2', 'FG2M', 'FG2A', 'FG2_PCT', 'NS_FG2_PCT', 'PLUSMINUS2']
        two_pt_df = two_pt_df.drop(['FG2M', 'FG2A', 'TEAM_ID', 'TEAM_ABBREVIATION', 'GP', 'G'], axis=1)


        resp2 = requests.get(three_pt_url, headers=headers, timeout=5).json()
        three_pt_results = resp2['resultSets']  # where this dict will be the json
        three_pt_cols = three_pt_results[0]['headers']
        three_pt_cols.remove('TEAM_NAME')
        three_pt_data = three_pt_results[0]['rowSet']
        three_pt_names = [team.pop(1) for team in three_pt_data]
        three_pt_df = pd.DataFrame(index=three_pt_names, data=three_pt_data, columns=three_pt_cols)
        three_pt_df.columns = ['TEAM_ID', 'TEAM_ABBREVIATION', 'GP', 'G',
                                  'FREQ3', 'FG3M', 'FG3A', 'FG3_PCT', 'NS_FG3_PCT', 'PLUSMINUS3']
        three_pt_df = three_pt_df.drop(['FG3M', 'FG3A'], axis=1)

        combined_defense_df = pd.concat([two_pt_df, three_pt_df], axis=1)

        return combined_defense_df

    except Exception as e:
        print(e)


def bref_box(url):

    resp = remove_comment_tags(url)
    soup = bs.BeautifulSoup(resp, 'lxml')

    try:
        game_div = soup.find('div', {'class': 'game_summaries'})

        games = [game for game in game_div.find_all('table', {'class':'teams'})]
        data = [tr for game in games for tr in game.tbody.find_all('tr')]
        teams = [td.get_text() for tr in data for td in tr.find_all('td', {'class': None})]
        scores = [td.get_text() for tr in data for td in tr.find_all('td', {'class': 'right'})][::2]

        df = pd.DataFrame()
        df['home_team'] = teams[1::2]
        df['home_score'] = scores[1::2]
        df['away_team'] = teams[::2]
        df['away_score'] = scores[::2]
        df['home_win'] = 0

        for row in range(df.shape[0]):
            if df['home_score'][row] > df['away_score'][row]:
                df['home_win'][row] = 1

        return df

    except Exception as e:
        print('Likely no games on this date')  # what error is this exactly?


def generate_season(season):

    start_date = '11-15-' + season[:4]  # add ~ a month to get 20 game threshold. *could* be done w/ timedelta
    start = datetime.datetime.strptime(start_date, '%m-%d-%Y')
    today = datetime.datetime.today()
    today = datetime.datetime.strftime(today, '%m-%d-%Y')  # better way to do, with split (?)

    if season == '2018-19':
        end = datetime.datetime.strptime(today, '%m-%d-%Y')
    else:
        end = '04-20-20' + season[-2:]  # latest game seemed to be around this time
        end = datetime.datetime.strptime(end, '%m-%d-%Y')

    date_range = end - start
    date_list = [end - datetime.timedelta(days=x) for x in range(1, date_range.days)]

    return date_list


def _get_data(date, season):
    ''' combines all the above, stores data in two dicts. Date is a single day '''

    curr_date = date.date().__str__()
    year = curr_date[:4]
    month = curr_date[5:7]
    day = curr_date[-2:]

    api_basic_url = f'https://stats.nba.com/stats/leaguedashteamstats?Conference=&DateFrom=&DateTo={curr_date}&Division=&GameScope=&GameSegment=&LastNGames=0&LeagueID=&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=&PaceAdjust=N&PerMode=Totals&Period=0&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season={season}&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&StarterBench=&TeamID=&TwoWay=&VsConference=&VsDivision=&'
    api_advanced_url = f'https://stats.nba.com/stats/leaguedashteamstats?Conference=&DateFrom=&DateTo={curr_date}&Division=&GameScope=&GameSegment=&LastNGames=0&LeagueID=&Location=&MeasureType=Advanced&Month=0&OpponentTeamID=0&Outcome=&PORound=&PaceAdjust=N&PerMode=Totals&Period=0&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season={season}&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&StarterBench=&TeamID=&TwoWay=&VsConference=&VsDivision=&'
    two_pt_url = f'https://stats.nba.com/stats/leaguedashptteamdefend?Conference=&DateFrom=&DateTo={curr_date}&DefenseCategory=2%20Pointers&Division=&GameSegment=&LastNGames=&LeagueID=00&Location=&Month=&OpponentTeamID=&Outcome=&PORound=&PerMode=Totals&Period=&Season={season}&SeasonSegment=&SeasonType=Regular+Season&TeamID=&VsConference=&VsDivision='
    three_pt_url = f'https://stats.nba.com/stats/leaguedashptteamdefend?Conference=&DateFrom=&DateTo={curr_date}&DefenseCategory=3%20Pointers&Division=&GameSegment=&LastNGames=&LeagueID=00&Location=&Month=&OpponentTeamID=&Outcome=&PORound=&PerMode=Totals&Period=&Season={season}&SeasonSegment=&SeasonType=Regular+Season&TeamID=&VsConference=&VsDivision='
    general_offense_url = f'https://stats.nba.com/stats/leaguedashteamptshot?CloseDefDistRange=&Conference=&DateFrom=&DateTo={curr_date}&Division=&DribbleRange=&GameSegment=&GeneralRange=&LastNGames=&LeagueID=00&Location=&Month=&OpponentTeamID=&Outcome=&PORound=&PerMode=Totals&Period=&Season={season}&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&ShotDistRange=&TeamID=&TouchTimeRange=&VsConference=&VsDivision='
    bref_url = f'https://www.basketball-reference.com/boxscores/?month={month}&day={day}&year={year}'


    basic_df = team_basic_box(api_basic_url)  
    advanced_df = team_advanced_box(api_advanced_url)  
    defense_df = gen_defense(two_pt_url, three_pt_url)
    offense_df = gen_offense(general_offense_url)

    try:
        basic_df = basic_df.loc[basic_df['GP'] >= 20]
        advanced_df = advanced_df.loc[advanced_df['GP'] >= 20]
        defense_df = defense_df.loc[defense_df['GP'] >= 20]
        offense_df = offense_df.loc[offense_df['GP'] >= 20]
        
        # make sure that at least 1 game for each team has been recorded. store list of dfs in dict
        if advanced_df.shape[0] == 30 and basic_df.shape[0] == 30 and defense_df.shape[0] == 30 and offense_df.shape[0] == 30:
            data_dict[curr_date] = [basic_df, advanced_df, defense_df, offense_df]

            score_df = (bref_box(bref_url))  
            score_dict[curr_date] = score_df

    except Exception as e:
        print(e)

    return

bad_urls = []

def get_data(q, season="2014-15"):
    while not q.empty():
        work = q.get()  # e.g. (0, "10-23-2019")
        
        try:
            _get_data(work[1], season)  # date, season
        except Exception as e:
            logging.error(e)
            bad_urls.append((work[1], season))  # maybe go back thru this at the end
            
        q.task_done()  # "Indicate that a formerly enqueued task is complete"
        
    return True


def main(season="2014-15", n_threads=16):
    date_list = generate_season(season)
    n_threads = min(n_threads, len(date_list))
    q = Queue()

    for i in range(len(date_list)):  # I creating/enqueueing here, does that work?
        q.put((i, date_list[i]))
        
    for i in range(n_threads): 
        t = threading.Thread(target=get_data, args=(q, season))
        t.setDaemon(True)  # main will exit even if all threads arent correctly completed
        t.start()
        
    q.join()


if __name__ == '__main__':
    main()
   

