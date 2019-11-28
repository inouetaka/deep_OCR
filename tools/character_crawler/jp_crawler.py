import requests
import time
from bs4 import BeautifulSoup


def jp_char():
    url_list = [
        "https://kanji.jitenon.jp/cat/joyo.html",
        "https://kanji.jitenon.jp/cat/namae.html"
    ]

    add_char = []
    for url in url_list:
        response = requests.get(url)
        bs = BeautifulSoup(response.text, 'html.parser')
        bs_table = bs.find_all('td', class_="normalbg")

        with open("./jp_char.txt", "a")as t:
            for i, text in enumerate(bs_table):
                if bs_table[i].text in add_char:
                    print("被り:",bs_table[i].text)
                else:
                    print(bs_table[i].text)
                    add_char.append(bs_table[i].text)
                    t.write(bs_table[i].text)
        time.sleep(1)


#jp_char()

