from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import numpy as np
from six.moves import xrange

import data_preprocessor as dp
import model_helper as mp
import hyper_parameters as hp

vocab_path = 'cache/vocabulary.txt'
_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]


def run_chatbot():
    with tf.Session() as tf_session:
        FLAGS = hp.get_hyperparameter()
        model_instance = mp.make_seq2seq_model(tf_session, False, FLAGS, _buckets)
        model_instance.batch_size = 1
        vocab, rev_vocab = dp.load_vocabulary(vocab_path)

        sys.stdout.write("You >: ")
        sys.stdout.flush()
        sentence = sys.stdin.readline()
        while sentence:
            token_ids = dp.encode_test_sentence(tf.compat.as_bytes(sentence), vocab)
            bucket_id = min([b for b in xrange(len(_buckets)) if _buckets[b][0] > len(token_ids)])
            encoder_inputs, decoder_inputs, target_weights = model_instance.get_batch({bucket_id: [(token_ids, [])]}, bucket_id)
            _, _, output_logits = model_instance.step(tf_session, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)
            outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
            if dp.EOS_ID in outputs:
                outputs = outputs[:outputs.index(dp.EOS_ID)]

            print("Chatbot >: " + " ".join([tf.compat.as_str(rev_vocab[output]) for output in outputs]))
            print("You >: ", end="")
            sys.stdout.flush()
            sentence = sys.stdin.readline()


def main(_):
    run_chatbot()

if __name__ == "__main__":
    tf.app.run()
