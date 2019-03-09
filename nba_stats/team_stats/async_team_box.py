import requests
import bs4 as bs
import pandas as pd
import json
import numpy as np
import datetime
from collections import defaultdict
import asyncio

def make_hash():
    return defaultdict(make_hash)


data_dict = make_hash()
score_dict = make_hash()

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

headers = {

    'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML like Gecko) Chrome/72.0.3626.81 Safari/537.36',
}

def remove_comment_tags(url, timeout=10):
    '''takes an input url and returns the DOM tree w/o comment tags wrapping the tables'''
    resp = requests.get(url, timeout=timeout)
    no_open_tag = resp.text.replace("""<!--\n   <div class="table_outer_container">""",
                                    """<div class="table_outer_container">""")
    no_close_tag = no_open_tag.replace("""</div>\n-->""","</div>")

    return no_close_tag


async def team_basic_box(url, filter_=True):
    ''' Scrapes NBA api for basic box score data'''
    try:
        resp = requests.get(url, headers=headers,timeout=10)
        resp.raise_for_status()
        stats = resp.json()
        col_names = stats['resultSets'][0]['headers']
        col_names.remove('TEAM_NAME')
        data = stats['resultSets'][0]['rowSet']
        team_names = [team.pop(1) for team in data]
        df = pd.DataFrame(index=team_names, data=data, columns=col_names)

        if filter_ is True:  # use team id to map between DFs
            keep_cols = df.columns.to_series()[['TEAM_ID', 'GP', 'PTS', 'AST', 'REB', 'STL', 'BLK']]
            df = df[keep_cols]

        return df

    except Exception as e:
        print(e)



async def team_advanced_box(url, filter_=True):
    ''' Scrapes NBA api for advanced box score data'''
    try:
        resp = requests.get(url, headers=headers,timeout=10)
        resp.raise_for_status()
        stats = resp.json()
        col_names = stats['resultSets'][0]['headers']
        col_names.remove('TEAM_NAME')
        data = stats['resultSets'][0]['rowSet']
        team_names = [team.pop(1) for team in data]
        df = pd.DataFrame(index=team_names, data=data, columns=col_names)

        if filter_ is True:
            keep_cols = df.columns.to_series()[['TEAM_ID', 'GP', 'W_PCT', 'OFF_RATING',
                                                'DEF_RATING', 'AST_TO', 'OREB_PCT', 'DREB_PCT', 'TS_PCT']]
            df = df[keep_cols]

        return df

    except Exception as e:
        print(e)




async def bref_box(url):
    ''' scrapes bref for daily box score data (teams and scores) '''
    resp = remove_comment_tags(url)
    resp.raise_for_status()
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
        print(e)
        print('Likely no games on this date')


def generate_season(season):
    ''' creates date format in prep for data consolidation'''
    start_date = '12-01-' + season[:4]  # add ~ a month to get 20 game threshold
    start = datetime.datetime.strptime(start_date, '%m-%d-%Y')
    today = datetime.datetime.today()
    today = datetime.datetime.strftime(today, '%m-%d-%Y')

    if season == '2018-19':
        end = datetime.datetime.strptime(today, '%m-%d-%Y')
    else:  # this needs to be better
        end = '04-15-20' + season[-2:]  # latest game seemed to be around this time
        end = datetime.datetime.strptime(end, '%m-%d-%Y')

    date_range = end - start
    date_list = [end - datetime.timedelta(days=x) for x in range(1, date_range.days)]
    return date_list


async def get_data(date, season):
    ''' consolidates data, stores in a dict'''
    curr_date = date.date().__str__()
    year = curr_date[:4]
    month = curr_date[5:7]
    day = curr_date[-2:]

    api_basic_url = f'https://stats.nba.com/stats/leaguedashteamstats?Conference=&DateFrom=&DateTo={curr_date}&Division=&GameScope=&GameSegment=&LastNGames=0&LeagueID=&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=&PaceAdjust=N&PerMode=Totals&Period=0&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season={season}&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&StarterBench=&TeamID=&TwoWay=&VsConference=&VsDivision=&'
    api_advanced_url = f'https://stats.nba.com/stats/leaguedashteamstats?Conference=&DateFrom=&DateTo={curr_date}&Division=&GameScope=&GameSegment=&LastNGames=0&LeagueID=&Location=&MeasureType=Advanced&Month=0&OpponentTeamID=0&Outcome=&PORound=&PaceAdjust=N&PerMode=Totals&Period=0&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season={season}&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&StarterBench=&TeamID=&TwoWay=&VsConference=&VsDivision=&'
    bref_url = f'https://www.basketball-reference.com/boxscores/?month={month}&day={day}&year={year}'

    basic_df = await team_basic_box(api_basic_url) 
    advanced_df = await team_advanced_box(api_advanced_url)  
    basic_df = basic_df.loc[basic_df['GP'] >= 20]
    advanced_df = advanced_df.loc[advanced_df['GP'] >= 20]

    if advanced_df.shape[0] == 30 and basic_df.shape[0] == 30:
        data_dict[curr_date]['advanced'] = advanced_df
        data_dict[curr_date]['basic'] = basic_df

        try:
            df = await (bref_box(bref_url))  # i think if an async function calls another it needs an await?
            score_dict[curr_date] = df

        except Exception as e:
            print(f'No games played on {start}')


    print(season, 'completed!')
    return

async def main():
    ''' runs above asynchronously for an entire season '''
    season = '2018-19'
    date_list = generate_season(season)
    tasks = []
    for date in date_list:
        tasks.append(asyncio.create_task(get_data(date, season)))

    await asyncio.wait([task for task in tasks])


if __name__ == '__main__':
    try:
        loop = asyncio.get_event_loop()  # even loop runs tasks one after another
        loop.run_until_complete(main())
        # asyncio.run(main())

    except RuntimeError as e:  # should log dates here at least, potentially rerun
        print(e)

    finally:
        loop.close()
