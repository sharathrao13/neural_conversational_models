import data_preprocessor as prepros
import seq2seq_model
import tensorflow as tf

def make_seq2seq_model(session, forward_only, FLAGS, _buckets, model_directory):
    model = seq2seq_model.Seq2SeqModel(
            FLAGS.vocab_size,
            FLAGS.vocab_size, _buckets,
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
        prepros.prepare_dataset_encoded(FLAGS.vocab_size)
        session.run(tf.initialize_all_variables())
    return model