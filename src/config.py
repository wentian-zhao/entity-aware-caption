import os
os.environ['PATH'] += ':/usr/local/lib/jdk1.8.0_241/bin'

coco_caption_path = '/home/wentian/work/coco-caption-py3'
# coco_caption_path = '/home/zwt/work/coco-caption-py3'

data_dir = '../data'

object_and_face_dir = '../data/objects_and_faces'

# roberta_path = '../pretrained_model/roberta.base.tar.gz'
roberta_path = '../pretrained_model/roberta.large.tar.gz'

image_dirs = {
    'goodnews': r"/media/wentian/sdb1/BaiduNetdiskDownload/215629229_entalent/news_image_captioning/data/goodnews/images_processed",
    'nytimes': r"/media/wentian/sdb1/BaiduNetdiskDownload/215629229_entalent/news_image_captioning/data/nytimes/images_processed"
}

_caption_preprocessed_dir = r'/media/wentian/nvme1/article_data/caption_preprocessed'
caption_processed_dirs = {
    'goodnews': os.path.join(_caption_preprocessed_dir, 'goodnews'),
    'nytimes': os.path.join(_caption_preprocessed_dir, 'nytimes')
}

_article_preprocessed_dir = r'/media/wentian/nvme1/article_data/preprocessed'
article_processed_dirs = {
    'goodnews': os.path.join(_article_preprocessed_dir, 'goodnews'),
    'nytimes': os.path.join(_article_preprocessed_dir, 'nytimes')
}