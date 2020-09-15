from flickrapi import FlickrAPI
from urllib.request import urlretrieve
from pprint import pprint
import os,time,sys

#info about API_key
KEY = os.environ['ANIMALCLASSIFIER_KEY']
SECRET_KEY = os.environ['ANIMALCLASSIFIER_SECRET_KEY']
WAITTIME = 1

searchText = sys.argv[1]
saveDir = "./" + searchText

flickr = FlickrAPI(KEY,SECRET_KEY,format = 'parsed-json')
result = flickr.photos.search(
    text = animalName,
    per_page = 400,
    media = 'photos',
    sort = 'relevance',
    safe_search = 1,
    extras = 'url_q, licence'
)

photos = result['photos']

if not os.path.exists(saveDir):
    ls.mkdir(saveDir)

for i,photo in enumerate(photos['photo']):
    url_q = photo['url_q']
    filepath = saveDir + '/' + photo['id'] + '.jpg'
    if os.path.exists(filepath):continue
    urlretrieve(url_q,filepath)
    time.sleep(WAITTIME)

