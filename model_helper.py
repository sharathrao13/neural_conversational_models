import preprocessing.data_preprocessor as dp
import models.seq2seq_model as seq2seq_model
import tensorflow as tf

dataset_file = 'data/Movie_Dataset'

def make_seq2seq_model(session, forward_only, FLAGS, buckets, model_directory):
    model = seq2seq_model.Seq2SeqModel(
            FLAGS.vocab_size,
            FLAGS.vocab_size, buckets,
            FLAGS.size,
            FLAGS.num_layers,
            FLAGS.gradients_clip,
            FLAGS.batch_size,
            FLAGS.learning_rate,
            FLAGS.learning_rate_decay,
            use_lstm=FLAGS.use_lstm,
            forward_only=forward_only)

    checkpoint= tf.train.get_checkpoint_state(model_directory)
    if checkpoint:
        print("Loading the model from checkpoint %s" % checkpoint.model_checkpoint_path)
        model.saver.restore(session, str(checkpoint.model_checkpoint_path))
    else:
        print("No checkpoints found. Starting with new model ...")
        dp.prepare_dataset_encoded(dataset_file, FLAGS.vocab_size)
        session.run(tf.initialize_all_variables())
    return model
