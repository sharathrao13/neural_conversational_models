from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import sys
import numpy as np

from six.moves import xrange
import model_helper as mp
import hyper_parameters as hp
import preprocessing.data_helper as data_helper
import tensorflow as tf

vocab_path = 'cache/vocabulary.txt'
dataset_file = 'data/Movie_Dataset'

_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]
model_directory = "./models/"
checkpoint_name = "ncm.ckpt"
steps_per_checkpoint = 25000

def train_chatbot():
    FLAGS = hp.get_hyperparameter()

    print("Training the chatbot now.\n The hyperparameters are:")
    print(type(FLAGS))
    print(FLAGS)

    # Avoids using all the GPUs available
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as tf_session:

        model_instance = mp.make_seq2seq_model(tf_session, False, FLAGS,_buckets, model_directory)

        print("Reading development and training data")
        validation_set = data_helper.read_data(en_dev, fr_dev)
        training_set = data_helper.read_data(en_train, fr_train)
        train_bucket_sizes = [len(training_set[b]) for b in xrange(len(_buckets))]
        train_total_size = float(sum(train_bucket_sizes))

        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size for i in
                               xrange(len(train_bucket_sizes))]

        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []

        print("Training begins now ...")
        while True:

            # The below code is based on translate.py in the older version of tensorflow
            ran_num = np.random.random_sample()
            bucket_id = min([i for i in xrange(len(train_buckets_scale)) if train_buckets_scale[i] > ran_num])

            encoder_inputs, decoder_inputs, target_weights = model_instance.get_batch(training_set, bucket_id)
            _, step_loss, _ = model_instance.step(tf_session, encoder_inputs, decoder_inputs, target_weights, bucket_id,
                                                 False)
            loss += step_loss / steps_per_checkpoint
            current_step += 1

            if current_step % steps_per_checkpoint == 0:
                perplexity = get_perplexity(loss)
                message = "global step %d learning rate %.4f step-time %.2f perplexity %.2f" % (
                model_instance.global_step.eval(), model_instance.learning_rate.eval(), step_time, perplexity)
                print(message)
                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    tf_session.run(model_instance.learning_rate_decay_op)
                previous_losses.append(loss)
                create_checkpoint(model_instance, tf_session)
                run_cross_validation(validation_set, tf_session, _buckets, model_instance)
                loss = 0.0


def create_checkpoint(seq2seq_model, tf_session):
    checkpoint_path = os.path.join(model_directory, checkpoint_name)
    seq2seq_model.saver.save(tf_session, checkpoint_path, global_step=seq2seq_model.global_step)


def run_cross_validation(validation_set, tf_session, buckets, model_instance):
    for bucket_id in xrange(len(buckets)):
        if len(validation_set[bucket_id]) == 0:
            print("  eval: empty bucket %d" % (bucket_id))
            continue
        encoder_inputs, decoder_inputs, target_weights = model_instance.get_batch(validation_set, bucket_id)
        _, cv_loss, _ = model_instance.step(tf_session, encoder_inputs, decoder_inputs, target_weights, bucket_id,
                                           True)
        cv_perplexity = get_perplexity(cv_loss)
        message = "Cross Validation: Bucket %d Perplexity %.2f" % (bucket_id, cv_perplexity)
        print(message)
        sys.stdout.flush()


def get_perplexity(loss):
    if loss < 400:
        perplexity = math.exp(loss)
    else:
        perplexity = float('inf')
    return perplexity


def main(_):
    train_chatbot()


if __name__ == "__main__":
    tf.app.run()
