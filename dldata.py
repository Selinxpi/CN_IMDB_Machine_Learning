import os
import shutil

import tensorflow as tf

url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

dataset = tf.keras.utils.get_file("aclImbd_v1", url, untar=True, cache_dir=".", cache_subdir="")

dataset_dir = os.path.jpin(os.path.dirname(dataset), 'aclImdb')
os.listdir(dataset_dir)
train_dir = os.path.join(dataset_dir, 'train')
os.listdir(train_dir)

remove_dir = os.path.join(train_dir, 'unsup')

shutil.rmtree(remove_dir)