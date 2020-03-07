import requests
import time
import urllib
from bs4 import BeautifulSoup

start_url = "https://en.wikipedia.org/wiki/Special:Random"
target_url = "https://en.wikipedia.org/wiki/Philosophy"

def continue_crawl(search_history, target_url, max_steps=25):
    if search_history[-1] == target_url:
        print("the last item is the target!")
        return False
    elif len(search_history) > max_steps:
        print("there are 25 items in the history list!")
        return False
    elif search_history[-1] in search_history[:-1]:
        print("there is a loop in the history list!")
        return False
    else:
        return True


def find_first_link(url):
    response = requests.get(url)
    html = response.text
    soup = BeautifulSoup(html,"html.parser")
    content_div = soup.find(id="mv-content-text").find(class_="mv-parser-output")
    article_link = None
    for element in content_div.find_all("p", recursive=False):
    	if element.find("a",recursive=False):
    		article_link = element.find("a",recoursive=False).get('href')
    		break
    if not article_link:
    	return
    first_link = urllib.parse.urljoin('https://en.wikipedia.org/', article_link)
    return first_link


article_chain = [start_url]

while continue_crawl(article_chain, target_url):
    print(article_chain[-1])

    first_link = find_first_link(article_chain[-1])
    if not first_link:
    	print("We've arrived at an article with no links")
    	break

    article_chain.append(first_link)
    time.sleep(2)
