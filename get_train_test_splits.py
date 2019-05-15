from __future__ import division
from __future__ import print_function
import codecs
import os
import numpy as np

import time
import tensorflow as tf

from utils import *
from metrics import *
from models import GCN_Align

# Set random seed
seed = 12306
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('lang', 'zh_en', 'Dataset string.')  # 'zh_en', 'ja_en', 'fr_en'
flags.DEFINE_float('learning_rate', 20, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 100, 'Number of epochs to train.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('gamma', 3.0, 'Hyper-parameter for margin based loss.')
flags.DEFINE_integer('k', 5, 'Number of negative samples for each positive seed.')
flags.DEFINE_float('beta', 0.9, 'Weight for structure embeddings.')
flags.DEFINE_integer('se_dim', 200, 'Dimension for SE.')
flags.DEFINE_integer('ae_dim', 100, 'Dimension for AE.')
flags.DEFINE_integer('seed', 3, 'Proportion of seeds, 3 means 30%')

# Load data
adj, ae_input, train, test = load_data(FLAGS.lang)


label_dir = '/home/mareike/PycharmProjects/kg/GCN-Align/data/preprocessing/{}'.format(FLAGS.lang)
out_dir = '/home/mareike/PycharmProjects/kg/GCN-Align/data/dicts/{}'.format(FLAGS.lang)

src_lang = 'zh'
trg_lang = 'en'


def get_idx2label(fname):
    with codecs.open(fname, 'r', 'utf-8') as f:
        lines = f.readlines()

    idx2label = {int(line.strip().split()[0]): ':::'.join(line.strip().split()[1:]) for line in lines}

    return idx2label

idx2label_src = get_idx2label(os.path.join(label_dir, 'labels_{}.txt.tokenized'.format(src_lang)))
idx2label_trg = get_idx2label(os.path.join(label_dir, 'labels_{}.txt.tokenized'.format(trg_lang)))

def write_seeds(idx2label_src, idx2label_trg, seeds, outname):
    with codecs.open(outname , 'w', 'utf-8') as f:
        for pair in seeds:
            src = pair[0]
            trg = pair[1]

            f.write('{} {}\n'.format(idx2label_src[src], idx2label_trg[trg]))
    f.close()

#for elm in test:
#    if elm[0] not in idx2label_src.keys():
#        print(elm[0])
write_seeds(idx2label_src, idx2label_trg, train, os.path.join(out_dir, 'jape.{}-{}.train.txt'.format(src_lang, trg_lang)))
write_seeds(idx2label_src, idx2label_trg, test, os.path.join(out_dir, 'jape.{}-{}.test.txt'.format(src_lang, trg_lang)))

train_flipped = train[:,[1,0]]
test_flipped = [(pair[1], pair[0]) for pair in test]

write_seeds(idx2label_trg, idx2label_src, train_flipped, os.path.join(out_dir, 'jape.{}-{}.train.txt'.format(trg_lang, src_lang)))
write_seeds(idx2label_trg, idx2label_src, test_flipped, os.path.join(out_dir, 'jape.{}-{}.test.txt'.format(trg_lang, src_lang)))

