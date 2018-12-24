import bs4 as bs
import collections
import pandas as pd
from urllib.request import urlopen
import re
import time
import functools
import numpy as np
pd.options.mode.chained_assignment = None

from utils import _gen_dataframe

class SingleSeasonStats(): 
    def __init__(self, year):
        self.year = year
        self.all_nba_dict = {}
        self.CACHE_FLAG = False

    @functools.lru_cache(maxsize=32)
    def basic_stats(self): 
        '''
        Retrieves all players and their basic stats DF for given season
        '''
        basic_url = 'https://www.basketball-reference.com/leagues/NBA_' + str(self.year) + '_per_game.html'
        basic_df = _gen_dataframe(basic_url)
        
        return basic_df

    @functools.lru_cache(maxsize=32)
    def adv_stats(self):
        """
        Generate advanced stats DF for a player, inlcuding ORTG, DRTG, Net 
        """
        
        adv_url = 'https://www.basketball-reference.com/leagues/NBA_' + str(self.year) + '_advanced.html'
        per_poss_url = 'https://www.basketball-reference.com/leagues/NBA_' + str(self.year) + '_per_poss.html'
        
        adv_df = _gen_dataframe(adv_url)
        all_per100 = _gen_dataframe(per_poss_url)

        ortg = all_per100['ORtg'].replace('', np.nan)
        drtg = all_per100['DRtg'].replace('', np.nan) 
        adv_df['ORtg'] = ortg
        adv_df['DRtg'] = drtg
        adv_df.drop('x', axis=1, inplace=True)

        adv_df.loc[:, 'ORtg'].replace('', np.nan) 
        adv_df.dropna(inplace = True)
        adv_df['Net Rtg'] = adv_df['ORtg'].astype(int) - adv_df['DRtg'].astype(int)
    
        return adv_df

    @functools.lru_cache(maxsize=32)
    def combine(self): #if really wanted could set year here (as w/ others) and reset inside. Has to be a better way
        '''
        Tries to combine adv and basic DFs. Otherwise returns just basic
        '''

        basic_df = self.basic_stats()
        basic_df = basic_df.drop(basic_df.columns.to_series()['Pos':'DRB'], axis=1)

        try:
            adv_df = self.adv_stats()
            com_df = basic_df.join(adv_df)
            self._add_label(com_df)
            com_df = com_df.drop(com_df.columns.to_series()['Pos':'MP'], axis=1)
            
            return com_df

        except Exception as e:
            print(e)
            self._add_label(basic_df)
            
            return basic_df
        
        
    def _gen_all_nba(self):
        '''
        Generates, returns, and caches the all-nba player list of input year
        '''
        url = 'https://www.basketball-reference.com/leagues/NBA_' + str(self.year)+'.html'
        soup = bs.BeautifulSoup(urlopen(url), 'lxml')
        all_nba = soup.find('div', id='all_honors')
        players = re.findall(r"'>(\w*[-\s]\w*['\s-]*\w*)", str(all_nba))
        
        self.all_nba_dict[self.year] = players

    def _add_label(self, df):
        '''
        returns new df with binary col indicating whether player made all-nba that year by comparing to cache
        of all-nba players
        '''
        if self.CACHE_FLAG is False:
            self._gen_all_nba()

        df['all_nba'] = 0
        for player in df.index:
            if player in self.all_nba_dict[self.year]:
                df['all_nba'].loc[player] = 1
        
        self.CACHE_FLAG = True
        return df


class MultiSeasonStats(SingleSeasonStats):
    def __init__(self, start_year, year):
        super().__init__(year)
        self.start_year = start_year
        
    @functools.lru_cache(maxsize=64)
    def multi_season(self, start_year=None):
        if start_year is None:
            start_year = self.start_year
        
        season_range = range(start_year, self.year+1)
        pool = ThreadPool(25)
        df_container = pool.map(self.combine_multi, [season for season in season_range])
        final_df = pd.concat(df_container)
        pool.close

        return final_df


    def _combine_multi(self, start_year=None):
        if start_year is None:
            start_year = self.start_year
        
        basic_url = 'https://www.basketball-reference.com/leagues/NBA_' + str(start_year) + '_per_game.html'
        adv_url = 'https://www.basketball-reference.com/leagues/NBA_' + str(start_year) + '_advanced.html'
        per_poss_url = 'https://www.basketball-reference.com/leagues/NBA_' + str(start_year) + '_per_poss.html'

        basic_df = _gen_dataframe(basic_url)
        basic_df = basic_df.drop(basic_df.columns.to_series()['Pos':'DRB'], axis=1)

        adv_df = _gen_dataframe(adv_url)
        per_poss_df = _gen_dataframe(per_poss_url)
    
        ortg = per_poss_df['ORtg'].replace('', np.nan)
        drtg = per_poss_df['DRtg'].replace('', np.nan) 
        adv_df['ORtg'] = ortg
        adv_df['DRtg'] = drtg
        adv_df.drop('x', axis=1, inplace=True)

        adv_df.loc[:, 'ORtg'].replace('', np.nan) 
        adv_df.dropna(inplace = True)
        adv_df['Net Rtg'] = adv_df['ORtg'].astype(int) - adv_df['DRtg'].astype(int)
        
        com_df = basic_df.join(adv_df)
        com_df = com_df.drop(com_df.columns.to_series()['Pos':'MP'], axis=1)
        self._add_label(com_df)

        return com_df


        

