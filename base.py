import bs4 as bs
import pandas as pd
import requests
import re
import time
import functools
import numpy as np
pd.options.mode.chained_assignment = None

from utils import _gen_dataframe, remove_comment_tags

class SingleSeasonStats(): 
    def __init__(self, year):
        self.year = year
        self.all_nba_dict = {}
        if datetime.now().month <10:
            self.current_year = datetime.now().year
        else:
            self.current_year = datetime.now().year + 1
        
    def _gen_dataframe(self, url): 
        '''
        Boilerplate DF generator
        '''
        cols = []
        soup = bs.BeautifulSoup(requests.get(url).text, 'lxml')
        table = soup.find('div', class_='table_outer_container')

        for th in table.thead.find_all('th'):
            if th.get_text() == '\xa0':
                cols.append('x')
            else:
                cols.append(th.get_text())

        n_cols = len(table.thead.find_all('th')) - 1
        data = [td.get_text() for tr in table.tbody.find_all('tr', class_='full_table') for td in tr.find_all('td')]


        for i in range(len(data)):
            try:
                data[i] = float(data[i])

            except Exception as e: #cant convert ... to float
                data[i] = data[i]

        data = [data[i:i+n_cols] for i in range(0, len(data), n_cols)]

        all_players = [sublist.pop(0).replace('*', '') for sublist in data]

        cols = cols[2:]
        df = pd.DataFrame(index=all_players, data=data, columns=cols)

        return df

    @functools.lru_cache(maxsize=32)
    def basic_stats(self): 
        '''
        Retrieves all players and their basic stats DF for given season
        '''
        basic_url = 'https://www.basketball-reference.com/leagues/NBA_' + str(self.year) + '_per_game.html'
        basic_df = self._gen_dataframe(basic_url)
        
        return basic_df

    @functools.lru_cache(maxsize=32)
    def adv_stats(self):
        """
        Generate advanced stats DF for a player, inlcuding ORTG, DRTG, Net 
        """
        
        adv_url = 'https://www.basketball-reference.com/leagues/NBA_' + str(self.year) + '_advanced.html'
        per_poss_url = 'https://www.basketball-reference.com/leagues/NBA_' + str(self.year) + '_per_poss.html'
        
        adv_df = self._gen_dataframe(adv_url)
        all_per100 = self._gen_dataframe(per_poss_url)

        ortg = all_per100['ORtg'].replace('', np.nan)
        drtg = all_per100['DRtg'].replace('', np.nan) 
        adv_df['ORtg'] = ortg
        adv_df['DRtg'] = drtg
        adv_df.drop('x', axis=1, inplace=True)

        adv_df.loc[:, 'ORtg'].replace('', np.nan) 
        adv_df.dropna(inplace = True)
        adv_df['Net Rtg'] = adv_df['ORtg'] - adv_df['DRtg']
        

    
        return adv_df

    @functools.lru_cache(maxsize=32)
    def combine(self, drop_no_contract=True): #if really wanted could set year here (as w/ others) and reset inside. Has to be a better way
        '''
        Tries to combine adv and basic DFs. Otherwise returns just basic
        '''

        basic_df = self.basic_stats()
        basic_df = basic_df.drop(basic_df.columns.to_series()['Pos':'DRB'], axis=1)

        try:
            adv_df = self.adv_stats()
            com_df = basic_df.join(adv_df)
            com_df.drop(com_df.columns.to_series()['Pos':'G'], axis=1, inplace=True)
            
            com_df = com_df.join(self._gen_salaries())
            self._all_nba_label(com_df)
            
            if drop_no_contract is True:
                com_df.dropna(inplace=True)

            com_df.replace(np.nan, 'Not under contract')
            
            return com_df

        except Exception as e:
            print(e)
            self._all_nba_label(basic_df)
            
            return basic_df
        
    def _all_nba_label(self, df):
        '''
        returns new df with binary colum indicating whether player made all-nba that year.
        Compares takes from cache if function has been run previously
        '''
        if ~(self.year in self.all_nba_dict):
            url = 'https://www.basketball-reference.com/leagues/NBA_' + str(self.year)+'.html'
            soup = bs.BeautifulSoup(requests.get(url).text, 'lxml')
            all_nba = soup.find('div', id='all_honors')
            players = re.findall(r"'>(\w*[-\s]\w*['\s-]*\w*)", str(all_nba))

            self.all_nba_dict[self.year] = players

        df['all_nba'] = 'no'
        for player in df.index:
            if player in self.all_nba_dict[self.year]:
                df['all_nba'].loc[player] = 'yes'
        
        return df
    
    def _gen_salaries(self):
        resp = requests.get('https://www.basketball-reference.com/contracts/players.html').text
        soup = bs.BeautifulSoup(resp, 'lxml')
        table = soup.find('div', class_='table_outer_container')
        cols = [th.get_text() for th in table.thead.find('tr', class_=None).find_all('th')]
        cols = cols[2:]
        data = [td.get_text() for td in table.tbody.find_all('td')]
        data = [element.replace('$', '') for element in data]
        data = [element.replace(',', '') for element in data]

        data = [data[i:i+10] for i in range(0, len(data), 10)]
        players = [sublist.pop(0) for sublist in data]

        salary_df = pd.DataFrame(index=players, columns=cols, data=data)
        salary_df['current_salary'] = current_salary = salary_df['2018-19'].astype(int)
        if self.year == self.current_year:
            salary_df['contract_year'] = np.where(salary_df['2019-20'] == '', 'yes', 'no') #or however its returned.
            current_salary = salary_df[['current_salary', 'contract_year']]   #would need to change later addition to pd.concat as well.

        return current_salary
    
    def salary_comparison(self, X_name, top_n=100, annotation=1, rect_shape=[.25, 1.2, 1.75, 1.5]): #maybe have annotations default to some int value
        plt.style.use('ggplot')

        cmap = cm.get_cmap('plasma')
        ax1 = plt.axes(rect_shape)

        df = self.combine().sort_values('MP', ascending=False).head(top_n)
        X = np.asarray(df[X_name], dtype=np.float)
        y = np.asarray(df['current_salary'], dtype=np.float)
        y_range = range(0, 45, 5)
        cmap = cmap(minmax_scale(X)- .1)
        ax1.scatter(X, y, s=20, c=cmap)


        xticks = ax1.get_xticklabels() #get tick labels so we can adjust them a bit with .setp
        plt.setp(xticks, rotation=45)

        ax1.plot(np.unique(X), np.poly1d(np.polyfit(X, y, 1))(np.unique(X)))
        ax1.set(yticklabels=y_range, xlabel=X_name, ylabel='Salary (millions)', 
                title="{} vs. Salary".format(X_name))

        if annotation is not None:
            if isinstance(annotation, str):
                try:
                    player_idx = df.index.get_loc(annotation)
                    ax1.annotate(annotation, (X[player_idx], y[player_idx]), xytext=(-10,-10),
                                textcoords='offset pixels', size=8) #change style later

                except Exception as e:
                    print('{} is not in the sample. Did you enter the name correctly?'.format(e))

            if isinstance(annotation, int):
                df_top = df.sort_values(X_name, ascending=False).head(annotation)
                df_bottom = df.sort_values(X_name, ascending=True).head(annotation)
                names_top = [i for i in df_top.index][::-1]
                names_bottom = [i for i in df_bottom.index] #do i need to reverse this?
                array = np.array(df[X_name])
                X_top = (array.argsort()[-annotation:]) #do i need the astype?
                X_bottom = (array.argsort()[:annotation])
                new_names_top = []
                new_names_bottom = []

                for i in names_top:
                    name = i.split(' ')
                    new_names_top.append(name[0][0] + '. ' + name[1])

                for i in names_bottom:
                    name = i.split(' ')
                    new_names_bottom.append(name[0][0] + '. ' + name[1])

                array = np.array(df[X_name])
                array_sort = (array.argsort()[-3:]).astype(int)

                for i, x_idx in enumerate(X_top):
                    ax1.annotate(new_names_top[i], xy=(X[x_idx], y[x_idx]), xytext=(-10,-10),
                            arrowprops=dict(arrowstyle='->'), textcoords='offset pixels', size=8) #arrowprops not working as intended

                for i, x_idx in enumerate(X_bottom):
                    ax1.annotate(new_names_bottom[i], xy=(X[x_idx], y[x_idx]), xytext=(-10,-10),
                            arrowprops=dict(arrowstyle='->'), textcoords='offset pixels', size=8) #arrowprops not working as intended



class MultiSeasonStats(SingleSeasonStats):
    def __init__(self, start_year, year):
        super().__init__(year)
        self.start_year = start_year
        
    @functools.lru_cache(maxsize=64)
    def multi_season_stats(self, start_year=None):
        if start_year is None:
            start_year = self.start_year
        
        season_range = range(start_year, self.year+1)
        pool = ThreadPool(25)
        df_container = pool.map(self.combine_multi, [season for season in season_range])
        final_df = pd.concat(df_container)
        pool.close()

        return final_df


    def _combine_dataframes(self, start_year=None):
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


        

