class StatsGrabber(object):
    
    def __init__(self, year):
        self.year = year
        self.basic_url = 'https://www.basketball-reference.com/leagues/NBA_' + str(year) + '_per_game.html'
        self.adv_url = 'https://www.basketball-reference.com/leagues/NBA_' + str(year) + '_advanced.html'
        self.per_poss_url = 'https://www.basketball-reference.com/leagues/NBA_' + str(year) + '_per_poss.html'
    
    def _gen_dataframe(url):
        """
        Boilerplate function for taking a stats URL and creating a dataframe
        """
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

        all_players = [sublist.pop(0).replace('*', '') for sublist in data] #annoying

        cols = cols[2:]
        df = pd.DataFrame(index=all_players, data=data, columns=cols)
        df['MP'] = df['MP'].astype(float) #make sure this doesnt throw errors

        return df
        
    
    def _gen_basic(year): 
        '''
        Retrieves all players and their basic stats DF for given season
        '''
        basic_df = _gen_dataframe(self.basic_url)
        return basic_df
        
    
    def _gen_netrtg():
        '''
        Retrieves ORTG, DRTG from per100 stats so that Net column can be created (in wrapper())
        '''
        all_per100 = _gen_dataframe(self.per_poss_url)

        ortg = all_per100['ORtg'].replace('', np.nan)
        drtg = all_per100['DRtg'].replace('', np.nan)

        rating_df = pd.DataFrame()
        rating_df['ORtg'] = pd.Series(ortg)
        rating_df['DRtg'] = pd.Series(drtg)
        
        return rating_df
    
    
    def _gen_adv():
        """
        Generate advanced stats DF for a player, inlcuding ORTG, DRTG
        """
        adv = pd.DataFrame(_gen_dataframe(self.adv_url))
        net = pd.DataFrame(_gen_netrtg(self.per_poss_url))
        adv_df = pd.concat((adv, net), axis=1)
        adv_df['MP'] = adv_df['MP'].astype(int)

        return df
