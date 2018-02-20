import praw
import tablib
import pytesseract
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import io
import requests
from bs4 import BeautifulSoup
from PIL import Image

reddit = praw.Reddit(client_id='',
                     client_secret="", password='',
                     user_agent='', username='')

top100 = reddit.subreddit('all').hot(limit=100)
analyze = SentimentIntensityAnalyzer()

# Tesseract path option
pytesseract.pytesseract.tesseract_cmd = 'D:\TesseractOCR\Tesseract-OCR\Tesseract'

# dataset creation and loading
data = tablib.Dataset()
data.headers = ['Title', 'Intensity', 'Upvotes', 'Up_ratio']
titles = []
intensity = []
upvotes = []
Up_ratio = []
Imagetext = []

for submission in top100:

    # title feature, possible Levenshtein implementation eventually to catch intentionally mispelled company names
    titles.append(submission.title)
    # intensity feature
    IS = analyze.polarity_scores(submission.title)
    intensity.append(IS)
    # upvoting features
    upvotes.append(submission.ups)
    Up_ratio.append(submission.upvote_ratio)

    # image features
    # catching, conversion, and encapsulation
    try:
        test = requests.get(submission.url)
        test.raise_for_status()
    except requests.exceptions.Timeout:
        print("Timeout")
    except requests.exceptions.TooManyRedirects:
        print("Bad URL")
    except requests.exceptions.RequestException as e:
        print("Critical ", e)

    # Parsing HTML with BS4, Checking, and conversion
    if "/i.imgur.com/" in submission.title:
        pic = Image.open(submission.url)
        pictext = io.BytesIO(pytesseract.image_to_string(pic))
        Imagetext.append(pictext)
    else:
        Imagetext.append("n")
    print('\n')

data.append(titles)
data.append(intensity)
data.append(upvotes)
data.append(Up_ratio)
data.append(Imagetext)

# data is read into csv in binary
with open('D:\Coding Projects\HC_Bot\post_data.csv', 'wb') as p:
    p.write(data.csv)
