import bs4 as bs
import requests
import pandas as pd
import numpy

def _gen_dataframe(url): 
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
    data = [data[i:i+n_cols] for i in range(0, len(data), n_cols)]

    all_players = [sublist.pop(0).replace('*', '') for sublist in data]

    cols = cols[2:]
    df = pd.DataFrame(index=all_players, data=data, columns=cols)

    return df


def remove_comment_tags(url):
    """
    takes an input url and returns the DOM tree w/o comment tags wrapping the tables
    """
    resp = requests.get(url)

    #remove the opening comment tag
    no_open_tag = resp.text.replace("""<!--\n   <div class="table_outer_container">""",
                                    """<div class="table_outer_container">""")
    #remove closing comment tag
    no_close_tag = no_open_tag.replace("""</div>\n-->""","</div>")
    
    return no_close_tag
