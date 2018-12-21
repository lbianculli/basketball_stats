class BrefStats():    
    def __init__(self):
#         self.start = start
#         self.end = end
        

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

    
    def _gen_basic(self, year): 
        '''
        Retrieves all players and their basic stats DF for given season
        '''
        
        basic_url = 'https://www.basketball-reference.com/leagues/NBA_' + str(year) + '_per_game.html'
        basic_df = self._gen_dataframe(basic_url)
        
        return basic_df


    def _gen_netrtg(self, year):
        '''
        Retrieves ORTG, DRTG from per100 stats so that Net column can be created (in wrapper())
        '''
        
        per_poss_url = 'https://www.basketball-reference.com/leagues/NBA_' + str(year) + '_per_poss.'
        all_per100 = self._gen_dataframe(per_poss_url)

        ortg = all_per100['ORtg'].replace('', np.nan)
        drtg = all_per100['DRtg'].replace('', np.nan)

        rating_df = pd.DataFrame()
        rating_df['ORtg'] = pd.Series(ortg)
        rating_df['DRtg'] = pd.Series(drtg)

        return rating_df

    def _gen_adv(self, year):
        """
        Generate advanced stats DF for a player, inlcuding ORTG, DRTG
        """
        
        adv_url = 'https://www.basketball-reference.com/leagues/NBA_' + str(year) + '_advanced.html'
        adv = self._gen_dataframe(adv_url)
        net = self._gen_netrtg(year)
        adv_df = pd.concat((adv, net), axis=1)
        adv_df['MP'] = adv_df['MP'].astype(int)

        return adv_df
    
   
    @functools.lru_cache
    def _combine(self, year): 
    '''
    Tries to combine adv and basic DFs. Otherwise returns just basic
    '''
    basic_df = self._gen_basic(year)
    basic_df = basic_df.drop(basic_df.columns.to_series()['Pos':'DRB'], axis=1)

    try:

        adv_df = self._gen_adv(year)
        adv_df.drop('x', axis=1, inplace=True) 

        com_df = pd.concat((adv_df, basic_df), axis=1)

        com_df.loc[:, 'ORtg'].replace('', np.nan) 
        com_df.dropna(inplace = True)
        com_df['Net Rtg'] = com_df['ORtg'].astype(int) - com_df['DRtg'].astype(int)
        com_df['MP'] = com_df['MP'].astype(int)
        com_df = com_df.drop(com_df.columns.to_series()['Pos':'MP'], axis=1)
        _add_label(com_df, year)
        return com_df

    except Exception as e:
        print(e)
        _add_label(basic_df, year)
        return basic_df


def _add_label(self, df, year):
    '''
    returns new df with binary col indicating whether player made all-nba that year by comparing to cache
    of all-nba players
    '''
    df['all_nba'] = 0

    for player in df.index:
        if player in all_nba_dict[year]:
            df['all_nba'].loc[player] = 1

    return df



    def _gen_seasons(start, end):
        '''
        For now, this function will create list of ALL season links, and return ones in given range.
        Ideally, it should only generate the ones in the given range in the first place.
        '''
        soup = bs.BeautifulSoup(urlopen('https://www.basketball-reference.com/leagues/'), 'lxml')
        seasons_col = soup.find('table',  id='stats').find_all('tr', class_=None)[1:]
        anchors = season.find_all('th').find('a')
        season_links = [anchor['href'] for a in anchors]

        #does this work as intended?
        start_url = 'https://www.basketball-reference.com/leagues/NBA_' + str(start)
        end_url = 'https://www.basketball-reference.com/leagues/NBA_' + str(end)
        season_links = season_links[start:end]
        seasons = [season[-8:-4] for season in season_links]

        return seasons
    
        
    def pool_seasons(start, end, n_threads=25):
        '''
        Returns one combined DF for a range of seasons given start and end year
        '''
        season_range = _gen_seasons(start, end)
        p = ThreadPool(n_threads)
        df_container = p.map(_combine, [season for season in season_range])
        final_df = pd.concat(df_container)
        p.close
        
        return final_df
