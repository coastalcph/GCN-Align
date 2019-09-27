from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

from utils import *
from metrics import *
from models import GCN_Align
import io
import codecs

# Set random seed
seed = 12306
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('lang', 'ja_en', 'Dataset string.')  # 'zh_en', 'ja_en', 'fr_en'
flags.DEFINE_string('in_dir', 'data', 'Input directory.')
flags.DEFINE_float('learning_rate', 20, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 20, 'Number of epochs to train.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('gamma', 3.0, 'Hyper-parameter for margin based loss.')
flags.DEFINE_integer('k', 5, 'Number of negative samples for each positive seed.')
flags.DEFINE_float('beta', 0.9, 'Weight for structure embeddings.')
flags.DEFINE_integer('se_dim', 200, 'Dimension for SE.')
flags.DEFINE_integer('ae_dim', 100, 'Dimension for AE.')
flags.DEFINE_integer('seed', 3, 'Proportion of seeds, 3 means 30%')
flags.DEFINE_string('out_dir', '', 'Output directory.')
flags.DEFINE_string('dico_train', '', 'Train dict.')
flags.DEFINE_string('dico_test', '', 'Test dict.')

src_lang = FLAGS.lang.split('_')[-2]
trg_lang = FLAGS.lang.split('_')[-1]
# Load data
adj, _, train, test = load_data_fixed_traintestset(FLAGS.in_dir, FLAGS.dico_train, FLAGS.dico_test)

# Some preprocessing
support = [preprocess_adj(adj)]
num_supports = 1
model_func = GCN_Align
k = FLAGS.k
#e = ae_input[2][0]

def get_idx2label(fname):
    with codecs.open(fname, 'r', 'utf-8') as f:
        lines = f.readlines()
    idx2label = {int(line.strip().split('\t')[0]): line.strip().split('\t')[1] for line in lines if len(line.split()) > 1}
    return idx2label

idx2word_src = get_idx2label('{}/idx2label_{}.txt'.format(FLAGS.in_dir, src_lang))
idx2word_trg = get_idx2label('{}/idx2label_{}.txt'.format(FLAGS.in_dir, trg_lang))

# Define placeholders

ph_se = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.placeholder(tf.float32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder_with_default(0, shape=())
}

# Create model
#model_ae = model_func(ph_ae, input_dim=ae_input[2][1], output_dim=FLAGS.ae_dim, ILL=train, sparse_inputs=True, featureless=False, logging=True)
e= adj.shape[0]
model_se = model_func(ph_se, input_dim=e, output_dim=FLAGS.se_dim, ILL=train, sparse_inputs=False, featureless=True, logging=True)
# Initialize session
sess = tf.Session()

# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []

t = len(train)
L = np.ones((t, k)) * (train[:, 0].reshape((t, 1)))
neg_left = L.reshape((t * k,))
L = np.ones((t, k)) * (train[:, 1].reshape((t, 1)))
neg2_right = L.reshape((t * k,))

# Train model

for epoch in range(FLAGS.epochs):
    if epoch % 10 == 0:
        neg2_left = np.random.choice(e, t * k)
        neg_right = np.random.choice(e, t * k)
        #vec_ae_test = sess.run(model_ae.outputs, feed_dict=feed_dict_ae_test)
        #vec_se_test = sess.run(model_se.outputs, feed_dict=feed_dict_se_test)
        # get_hits(vec_ae, test)
        #print("SE")
        #get_hits(vec_se_test, test)
    # Construct feed dictionary
    #feed_dict_ae = construct_feed_dict(ae_input, support, ph_ae)
    #feed_dict_ae.update({ph_ae['dropout']: FLAGS.dropout})
    #feed_dict_ae.update({'neg_left:0': neg_left, 'neg_right:0': neg_right, 'neg2_left:0': neg2_left, 'neg2_right:0': neg2_right})
    feed_dict_se = construct_feed_dict(1.0, support, ph_se)
    feed_dict_se.update({ph_se['dropout']: FLAGS.dropout})
    feed_dict_se.update({'neg_left:0': neg_left, 'neg_right:0': neg_right, 'neg2_left:0': neg2_left, 'neg2_right:0': neg2_right})
    # Training step
    #outs_ae = sess.run([model_ae.opt_op, model_ae.loss], feed_dict=feed_dict_ae)
    outs_se = sess.run([model_se.opt_op, model_se.loss, model_se.outputs], feed_dict=feed_dict_se)
    #cost_val.append((outs_ae[1], outs_se[1]))
    cost_val.append((outs_se[1]))
    #if epoch%10==0:
        #print("SE train")
        #get_hits(outs_se[2], train)
        #print("SE test")
        #get_hits(outs_se[2], test)
    # Print results
    #print("Epoch:", '%04d' % (epoch + 1), "AE_train_loss=", "{:.5f}".format(outs_ae[1]), "SE_train_loss=", "{:.5f}".format(outs_se[1]))
    print("Epoch:", '%04d' % (epoch + 1), "SE_train_loss=",
          "{:.5f}".format(outs_se[1]))

print("Optimization Finished!")

# Testing
#feed_dict_ae_test = construct_feed_dict(ae_input, support, ph_ae)
feed_dict_se_test = construct_feed_dict(1.0, support, ph_se)
#vec_ae_test = sess.run(model_ae.outputs, feed_dict=feed_dict_ae_test)
vec_se_test = sess.run(model_se.outputs, feed_dict=feed_dict_se_test)


def export_embs(lang, embs, idx2word):
    # export embeddings
    fname = os.path.join(FLAGS.out_dir, 'gcnalign_embs_{}.txt'.format(lang))
    # text file
    c = 0
    with io.open(fname, 'w', encoding='utf-8') as f:
        f.write('{} {}\n'.format(embs.shape[0], embs.shape[1]))
        for idx, word in idx2word.items():
            c += 1

            f.write("{} {}\n".format(word, " ".join('%.5f' % x for x in embs[idx, :])))
    f.close()





export_embs(src_lang, vec_se_test, idx2word_src)
export_embs(trg_lang, vec_se_test, idx2word_trg)

# get_hits(vec_ae, test)
print('Number of test pairs: {}'.format(len(test)))
print("SE")
get_hits(vec_se_test, test)

#print("SE+AE")
#get_combine_hits(vec_se_test, vec_ae_test, FLAGS.beta, test)
