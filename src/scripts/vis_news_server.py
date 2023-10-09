import json
import os
import sys
import argparse

import flask
from flask import Flask, request, send_file

from pymongo import MongoClient

goodnews_image_path = '/media/wentian/sdb1/BaiduNetdiskDownload/215629229_entalent/news_image_captioning/data/goodnews/images_processed'
nytimes_image_path = '/media/wentian/sdb1/BaiduNetdiskDownload/215629229_entalent/news_image_captioning/data/nytimes/images_processed'

client = MongoClient(host='127.0.0.1', port=27017)
db_goodnews = client.goodnews
db_nytimes = client.nytimes

app = Flask(__name__)


@app.route('/images/goodnews/<image_id>')
def get_image_goodnews(image_id):
    image_filename = os.path.join(goodnews_image_path, f'{image_id}.jpg')
    return send_file(image_filename, mimetype='image/jpeg')


@app.route('/images/nytimes/<image_id>')
def get_image_nytimes(image_id):
    image_filename = os.path.join(nytimes_image_path, f'{image_id}.jpg')
    return send_file(image_filename, mimetype='image/jpeg')


@app.route('/goodnews/<image_id>')
def query_goodnews(image_id):
    image_filename = os.path.join(goodnews_image_path, f'{image_id}.jpg')

    article_id = image_id[:24]
    article = db_goodnews.articles.find_one({
        '_id': {'$eq': article_id},
    }, projection=['_id', 'context', 'images', 'web_url'])
    context = article['context']

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{image_id}</title>
</head>
<body>
    <img src="/images/goodnews/{image_id}"></img>
    </br>
    <p>{context}</p>
</body>
</html>
    """

    return html


index_file = r'/media/wentian/sdb1/work/news_caption_fairseq/data/nytimes/index_nytimes.json'
with open(index_file, 'r') as f:
    d = json.load(f)
    image_id_to_article = dict(i[:2] for i in d['test'])
article_file = r'/media/wentian/sdb1/work/news_caption_fairseq/data/nytimes/articles_nytimes.json'
with open(article_file, 'r') as f:
    d_article = json.load(f)


@app.route('/nytimes/<image_id>')
def query_nytimes(image_id):
    image_filename = os.path.join(goodnews_image_path, f'{image_id}.jpg')

    article_id = image_id_to_article[image_id]
    context = d_article[article_id]

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>nytimes {image_id}</title>
</head>
<body>
    <img src="/images/nytimes/{image_id}"></img>
    </br>
    <p>{context}</p>
</body>
</html>
    """

    return html


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', default=8000, type=int)
    args = parser.parse_args()
    app.run(debug=True, host='0.0.0.0', port=args.port)