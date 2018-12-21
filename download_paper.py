
# coding: utf-8

# In[1]:

import urllib.request
import re
import time
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
from selenium import webdriver


# In[2]:

base_url="https://www.kegg.jp/dbget-bin/www_bget?map05223"
paper_link_pattern="DOI:<a href=\".*>"
req = urllib.request.Request(base_url)
html = urllib.request.urlopen(req)
docs = html.read().decode('utf8')

url_list = list(set(re.findall(paper_link_pattern,docs)))


# In[4]:

urls=[]
for i,v in enumerate(url_list):
    urls.append(v[13:-20])
urls[0]


# In[5]:

keep = []
for u in urls:
    lengths=re.search(r'"', u).span()[0]
    keep.append(u[:lengths])
print(keep)


# In[6]:

urlss=[]
for uri in keep[1:]:
    browser=webdriver.Chrome()
    browser.get(uri)
    time.sleep(5)
    urlss.append(browser.current_url)
urlss


# In[22]:

pdfs="http://cancerres.aacrjournals.org/content/canres/68/13/4971.full.pdf"


# In[44]:

req = urllib.request.Request(urlss[1])
html = urllib.request.urlopen(req)
docs = html.read().decode('utf8')
#ind = docs.find(r".pdf")
#ind
docs


# In[26]:

r = requests.get(pdfs, stream=True)

with open('./%s.pdf'%pdfs[10:20], 'wb') as fd:
    for chunk in r.iter_content(100):
        fd.write(chunk)


# In[39]:

for i in range(len(urlss)):
    req = urllib.request.Request(urlss[i])
    html = urllib.request.urlopen(req)
    docs = html.read().decode('utf8')
    ind = docs.find(r".pdf")


# We have to access the website in campus otherwise it would be expensive to download the papers. So I used selenium and  chromedriver.
# 

# Then we need to find the download link inside the keep list
