import tensorflow as tf

tf.app.flags.DEFINE_integer("batch_size", 64, "Batch size to use during training")
tf.app.flags.DEFINE_integer("size", 128, "Size of each model layer")
tf.app.flags.DEFINE_integer("num_layers", 2, "Number of layers in the model")
tf.app.flags.DEFINE_integer("vocab_size", 10000, "Vocabulary size")
tf.app.flags.DEFINE_boolean("use_lstm", False, "Use LSTM as cell")
tf.app.flags.DEFINE_float("learning_rate", 0.5, "Use LSTM as cell")
tf.app.flags.DEFINE_float("learning_rate_decay", 0.99, "Use LSTM as cell")
tf.app.flags.DEFINE_float("gradients_clip", 5.0, "Use LSTM as cell")

def get_hyperparameter():
    FLAGS = tf.app.flags.FLAGS
    return FLAGS