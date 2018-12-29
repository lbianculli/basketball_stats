def cache_single_rookie_year(year):
        
    url = 'https://www.basketball-reference.com/leagues/NBA_' + str(year) + '_per_game.html'
    init_soup = bs.BeautifulSoup(requests.get(url).text, 'lxml')
    table = init_soup.find('table', {'id': 'per_game_stats'})
    try:
        links = ['https://www.basketball-reference.com' + td.get('href') for tr in table.find_all('tr', class_='full_table') for td in tr.find('td')]
        player_names = [td.get_text() for tr in table.find_all('tr', class_='full_table') for td in tr.find('td')]
    except AttributeError as e:
        player_names = []
        links = []
        for tr in table.tbody.find_all('tr', class_='full_table')[:100]:
            for td in tr.find('td'):
                try:
                    links.append('https://www.basketball-reference.com' + re.findall(r'(?<=").*(?=")', str(td))[0])
                    player_names.append(re.findall(r'(?<=>).*(?=<)', str(td))[0])
                except Exception as e: #returning some empty lists
                    pass
            
    
    names_links = zip(player_names, links)

    for player_name, player_link in names_links:
        if player_name not in rookie_year_dict:
           
            player_resp = requests.get(player_link).text
            player_soup = bs.BeautifulSoup(player_resp, 'lxml')
            player_table = player_soup.find('table', {'id': 'per_game'})
            rookie_year = int(player_table.tbody.find('a').get_text()[:4]) + 1
            rookie_year_dict[player_name] = rookie_year
						
						
def pool_all_rookie_years():
    year_range = range(1976, 2019)
    pool = ThreadPool(25)
    pool.map(cache_single_rookie_year, [year for year in year_range])
    pool.close()
