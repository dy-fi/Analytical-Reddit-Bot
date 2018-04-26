from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import praw
import tablib
import tensorflow.contrib.learn as tf

#reddit object
reddit = praw.Reddit(client_id='',
                     client_secret="", password='',
                     user_agent='', username='')

#model specification
classifier = tf.TensorFlowLinearClassifier(n_classes=3)

#training set pipe objects and metadata
training = "D:\Coding Projects\HC_Bot\post_data.csv"
ds = tf.data.TextLineDataset(training).skip(1)
meta = ['Title', 'Intensity', 'Upvotes', 'Upvote Ratio', 'Image text','label']
defaults = [[""], [0.0], [0], [0.0], [""], [""]]

def parser(line):
#csv parsing and formatting
    fields = tf.decode_csv(line, defaults)
    features = dict(zip(meta,fields))

#isolate the label field and assign a variable to hold this
    label = features.pop('label')

    return features, label


ds = ds.map(parser)



