import pandas as pd
import bs4 as bs
from urllib.request import urlopen
import functools


class BrefStats():    
    def __init__(self, year):
        self.year = year
        self.basic_url = 'https://www.basketball-reference.com/leagues/NBA_' + str(year) + '_per_game.html'
        self.adv_url = 'https://www.basketball-reference.com/leagues/NBA_' + str(year) + '_advanced.html'
        self.per_poss_url = 'https://www.basketball-reference.com/leagues/NBA_' + str(year) + '_per_poss.html'

    def _gen_dataframe(self, url): 
        '''
        Boilerplate DF generator
        '''
        cols = []
        soup = bs.BeautifulSoup(urlopen(url), 'lxml')
        table = soup.find('div', class_='table_outer_container')

        for th in table.thead.find_all('th'):
            if th.get_text() == '\xa0':
                cols.append('x')
            else:
                cols.append(th.get_text())

        n_cols = len(table.thead.find_all('th')) - 1
        data = [td.get_text() for tr in table.tbody.find_all('tr', class_='full_table') for td in tr.find_all('td')]       
        data = [data[i:i+n_cols] for i in range(0, len(data), n_cols)]

        all_players = [sublist.pop(0).replace('*', '') for sublist in data]

        cols = cols[2:]
        df = pd.DataFrame(index=all_players, data=data, columns=cols)

        return df

    
    @functools.lru_cache(maxsize=32)
    def _gen_basic(self): 
        '''
        Retrieves all players and their basic stats DF for given season
        '''
        basic_df = self._gen_dataframe(self.basic_url)
        return basic_df


    def _gen_netrtg(self):
        '''
        Retrieves ORTG, DRTG from per100 stats so that Net column can be created (in wrapper())
        '''
        all_per100 = self._gen_dataframe(self.per_poss_url)

        ortg = all_per100['ORtg'].replace('', np.nan)
        drtg = all_per100['DRtg'].replace('', np.nan)

        rating_df = pd.DataFrame()
        rating_df['ORtg'] = pd.Series(ortg)
        rating_df['DRtg'] = pd.Series(drtg)

        return rating_df

    @functools.lru_cache(maxsize=32)
    def _gen_adv(self):
        """
        Generate advanced stats DF for a player, inlcuding ORTG, DRTG
        """
        
        adv = self._gen_dataframe(self.adv_url)
        net = self._gen_netrtg()
        adv_df = pd.concat((adv, net), axis=1)
        adv_df['MP'] = adv_df['MP'].astype(int)

        return adv_df
    

    @functools.lru_cache(maxsize=32)
    def _combine(self): 
        '''
        Tries to combine adv and basic DFs. Otherwise returns just basic
        '''
        basic_df = self._gen_basic()
        basic_df = basic_df.drop(basic_df.columns.to_series()['Pos':'DRB'], axis=1)

        try:

            adv_df = self._gen_adv()
            adv_df.drop('x', axis=1, inplace=True) 

            com_df = pd.concat((adv_df, basic_df), axis=1)

            com_df.loc[:, 'ORtg'].replace('', np.nan) 
            com_df.dropna(inplace = True)
            com_df['Net Rtg'] = com_df['ORtg'].astype(int) - com_df['DRtg'].astype(int)
            com_df['MP'] = com_df['MP'].astype(int)
   
            return com_df

        except Exception as e:
            print(e)
            return basic_df

    
    def _add_label(self):
        '''
        returns new df with binary col indicating whether player made all-nba that year by comparing to cache
        of all-nba players
        '''
        combined_df = _combine(year)
        soup = bs.BeautifulSoup(urlopen(self.basic_url), 'lxml') # I think this is the page with basic stats
        spans = [span.get_text() for span in soup.find_all('span')]
        df['all_nba'] = 0

        for player in df.index:
            if player in all_nba_dict[year]:
                df['all_nba'].loc[player] = 1

        df = df.drop(df.columns.to_series()['Pos':'MP'], axis=1)

        return df
