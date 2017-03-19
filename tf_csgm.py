from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# We disable pylint because we need python3 compatibility.
from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import zip  # pylint: disable=redefined-builtin

from tensorflow.python import shape
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest
import tensorflow as tf

import numpy as np

# TODO(ebrevdo): Remove once _linear is fully deprecated.
linear = rnn_cell._linear  # pylint: disable=protected-access


def _extract_argmax_and_embed(embedding, output_projection=None,
                              update_embedding=True):
    def loop_function(prev, _):
        if output_projection is not None:
            prev = nn_ops.xw_plus_b(
                    prev, output_projection[0], output_projection[1])
        prev_symbol = math_ops.argmax(prev, 1)
        # Note that gradients will not propagate through the second parameter of
        # embedding_lookup.
        emb_prev = embedding_ops.embedding_lookup(embedding, prev_symbol)
        if not update_embedding:
            emb_prev = array_ops.stop_gradient(emb_prev)
        return emb_prev

    return loop_function


def attention_decoder(decoder_inputs,
                      initial_state,
                      attention_states,
                      cell,
                      output_size=None,
                      num_heads=1,
                      loop_function=None,
                      dtype=None,
                      scope=None,
                      initial_state_attention=False):
    #print("Inside Attention Decoder")

    if not decoder_inputs:
        raise ValueError("Must provide at least 1 input to attention decoder.")
    if num_heads < 1:
        raise ValueError("With less than 1 heads, use a non-attention decoder.")
    if attention_states.get_shape()[2].value is None:
        raise ValueError("Shape[2] of attention_states must be known: %s"
                         % attention_states.get_shape())
    if output_size is None:
        output_size = cell.output_size

    with variable_scope.variable_scope(
                    scope or "attention_decoder", dtype=dtype) as scope:
        dtype = scope.dtype

        batch_size = array_ops.shape(decoder_inputs[0])[0]  # Needed for reshaping.
        attn_length = attention_states.get_shape()[1].value
        if attn_length is None:
            attn_length = shape(attention_states)[1]
        attn_size = attention_states.get_shape()[2].value

        # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
        hidden = array_ops.reshape(
                attention_states, [-1, attn_length, 1, attn_size])
        hidden_features = []
        v = []
        attention_vec_size = attn_size  # Size of query vectors for attention.
        for a in xrange(num_heads):
            k = variable_scope.get_variable("AttnW_%d" % a,
                                            [1, 1, attn_size, attention_vec_size])
            hidden_features.append(nn_ops.conv2d(hidden, k, [1, 1, 1, 1], "SAME"))
            v.append(
                    variable_scope.get_variable("AttnV_%d" % a, [attention_vec_size]))

        state = initial_state

        def attention(query):
            """Put attention masks on hidden using hidden_features and query."""
            ds = []  # Results of attention reads will be stored here.
            if nest.is_sequence(query):  # If the query is a tuple, flatten it.
                query_list = nest.flatten(query)
                for q in query_list:  # Check that ndims == 2 if specified.
                    ndims = q.get_shape().ndims
                    if ndims:
                        assert ndims == 2
                query = array_ops.concat(1, query_list)
            for a in xrange(num_heads):
                with variable_scope.variable_scope("Attention_%d" % a):
                    y = linear(query, attention_vec_size, True)
                    y = array_ops.reshape(y, [-1, 1, 1, attention_vec_size])
                    # Attention mask is a softmax of v^T * tanh(...).
                    s = math_ops.reduce_sum(
                            v[a] * math_ops.tanh(hidden_features[a] + y), [2, 3])
                    a = nn_ops.softmax(s)
                    # Now calculate the attention-weighted vector d.
                    d = math_ops.reduce_sum(
                            array_ops.reshape(a, [-1, attn_length, 1, 1]) * hidden,
                            [1, 2])
                    ds.append(array_ops.reshape(d, [-1, attn_size]))
            return ds

        outputs = []
        prev = None
        batch_attn_size = array_ops.pack([batch_size, attn_size])
        attns = [array_ops.zeros(batch_attn_size, dtype=dtype)
                 for _ in xrange(num_heads)]
        for a in attns:  # Ensure the second shape of attention vectors is set.
            a.set_shape([None, attn_size])
        if initial_state_attention:
            attns = attention(initial_state)
        for i, inp in enumerate(decoder_inputs):
            if i > 0:
                variable_scope.get_variable_scope().reuse_variables()
            # If loop_function is set, we use it instead of decoder_inputs.
            if loop_function is not None and prev is not None:
                with variable_scope.variable_scope("loop_function", reuse=True):
                    inp = loop_function(prev, i)
            # Merge input and previous attentions into one vector of the right size.
            input_size = inp.get_shape().with_rank(2)[1]
            if input_size.value is None:
                raise ValueError("Could not infer input size from input: %s" % inp.name)
            x = linear([inp] + attns, input_size, True)
            # Run the RNN.
            cell_output, state = cell(x, state)
            # Run the attention mechanism.
            if i == 0 and initial_state_attention:
                with variable_scope.variable_scope(variable_scope.get_variable_scope(),
                                                   reuse=True):
                    attns = attention(state)
            else:
                attns = attention(state)

            with variable_scope.variable_scope("AttnOutputProjection"):
                output = linear([cell_output] + attns, output_size, True)
            if loop_function is not None:
                prev = output
            outputs.append(output)

    print("The shape of the final output from the method attention_decoder {0},{1}".format(np.shape(outputs),
                                                                                           np.shape(state)))
    return outputs, state


def embedding_attention_decoder(decoder_inputs,
                                initial_state,
                                attention_states,
                                cell,
                                num_symbols,
                                embedding_size,
                                num_heads=1,
                                output_size=None,
                                output_projection=None,
                                feed_previous=False,
                                update_embedding_for_previous=True,
                                dtype=None,
                                scope=None,
                                initial_state_attention=False):
    #print("Inside Embedding Attention Decoder")

    if output_size is None:
        output_size = cell.output_size
    if output_projection is not None:
        proj_biases = ops.convert_to_tensor(output_projection[1], dtype=dtype)
        proj_biases.get_shape().assert_is_compatible_with([num_symbols])

    with variable_scope.variable_scope(
                    scope or "embedding_attention_decoder", dtype=dtype) as scope:

        embedding = variable_scope.get_variable("embedding",
                                                [num_symbols, embedding_size])
        loop_function = _extract_argmax_and_embed(
                embedding, output_projection,
                update_embedding_for_previous) if feed_previous else None
        emb_inp = [
            embedding_ops.embedding_lookup(embedding, i) for i in decoder_inputs]
        return attention_decoder(
                emb_inp,
                initial_state,
                attention_states,
                cell,
                output_size=output_size,
                num_heads=num_heads,
                loop_function=loop_function,
                initial_state_attention=initial_state_attention)


def embedding_attention_seq2seq(encoder_inputs,
                                context_inputs,
                                decoder_inputs,
                                cell,
                                num_encoder_symbols,
                                num_decoder_symbols,
                                embedding_size,
                                num_heads=1,
                                output_projection=None,
                                feed_previous=False,
                                dtype=None,
                                scope=None,
                                initial_state_attention=False):
    #print("Inside Method Embedding Attention Seq2Seq")

    #print("Shape of encoder input {0}".format(len(encoder_inputs)))
    #print("Shape of decoder input {0}".format(len(decoder_inputs)))
    #print("num_encoder_symbols = {0}".format(num_encoder_symbols))
    ###print("num_decoder_symbols {0}".format(num_decoder_symbols))
    #print("embedding_size {0}".format(embedding_size))
    print("output_projection {0}".format(output_projection))

    with variable_scope.variable_scope(
                    scope or "embedding_attention_seq2seq", dtype=dtype,reuse=None) as scope:
        dtype = scope.dtype
        
        with variable_scope.variable_scope("encoder") as scope2:# Encoder.
            encoder_cell = rnn_cell.EmbeddingWrapper(
                    cell, embedding_classes=num_encoder_symbols,
                    embedding_size=embedding_size)
            encoder_outputs, encoder_state = rnn.rnn(
                    encoder_cell, encoder_inputs, dtype=dtype)
            #print(type(encoder_outputs))
            #np.savetxt('encoder_output.txt', encoder_outputs)
            #np.savetxt('encoder_state.txt', encoder_state)

        with variable_scope.variable_scope("context") as scope3:
            context_cell = rnn_cell.EmbeddingWrapper(cell, embedding_classes=num_encoder_symbols,
                                                     embedding_size=embedding_size)
            context_outputs, context_state = rnn.rnn(
                    context_cell, context_inputs, dtype=dtype)

            #target = open("context_output.txt", 'w')
            #for out in context_outputs:
                #target.write(str(out))
                #target.write("\n")
            #target.close()

            #np.savetxt('context_output.txt', context_outputs)
            #np.savetxt('context_state.txt', context_state)

        #print("The dimension of context state {0}".format(context_state))
        #context_state = tf.Print(context_state,[context_state],message="Printing the context State")
        #context_state.eval()
        #tf.add(context_state,context_state).eval()

        #with tf.session as session_c:
            #session_c.run(context_state)

        #print("Inside method embedding_attention_seq2seq. Encoder Outputs {0} Encode State {1}".format(
                #np.shape(encoder_outputs), np.shape(encoder_state)))

        #encoder_outputs = tf.add(encoder_outputs, context_outputs)
        #encoder_state = tf.add(encoder_state, context_state)

        for i in range(len(encoder_outputs)):
            encoder_outputs[i] = tf.add(encoder_outputs[i],context_outputs[i])

        
        temp = []
        for i in range(len(encoder_state)):
            temp.append(tf.add(encoder_state[i],context_state[i]))
        encoder_state = tuple(temp)

        #print(type(encoder_outputs))
        #print(type(encoder_outputs[0]))
        #print(type(encoder_state))
        #print(encoder_state)


        print("Inside method embedding_attention_seq2seq. Encoder Outputs {0} Encode State {1}".format(
                np.shape(encoder_outputs), np.shape(encoder_state)))

        # First calculate a concatenation of encoder outputs to put attention on.
        top_states = [array_ops.reshape(e, [-1, 1, cell.output_size])
                      for e in encoder_outputs]
        attention_states = array_ops.concat(1, top_states)

        print("Attention States has been created of size {0}".format(np.shape(attention_states)))
        # Decoder.
        output_size = None
        if output_projection is None:
            cell = rnn_cell.OutputProjectionWrapper(cell, num_decoder_symbols)
            output_size = num_decoder_symbols
            print("The output size is {0}".format(output_size))

        if isinstance(feed_previous, bool):
            print("Number of heads {0}".format(num_heads))
            return embedding_attention_decoder(
                    decoder_inputs,
                    encoder_state,
                    attention_states,
                    cell,
                    num_decoder_symbols,
                    embedding_size,
                    num_heads=num_heads,
                    output_size=output_size,
                    output_projection=output_projection,
                    feed_previous=feed_previous,
                    initial_state_attention=initial_state_attention)

        # If feed_previous is a Tensor, we construct 2 graphs and use cond.
        def decoder(feed_previous_bool):
            reuse = None if feed_previous_bool else True
            with variable_scope.variable_scope(
                    variable_scope.get_variable_scope(), reuse=reuse) as scope:
                outputs, state = embedding_attention_decoder(
                        decoder_inputs,
                        encoder_state,
                        attention_states,
                        cell,
                        num_decoder_symbols,
                        embedding_size,
                        num_heads=num_heads,
                        output_size=output_size,
                        output_projection=output_projection,
                        feed_previous=feed_previous_bool,
                        update_embedding_for_previous=False,
                        initial_state_attention=initial_state_attention)
                state_list = [state]
                if nest.is_sequence(state):
                    state_list = nest.flatten(state)
                return outputs + state_list

        outputs_and_state = control_flow_ops.cond(feed_previous,
                                                  lambda: decoder(True),
                                                  lambda: decoder(False))
        outputs_len = len(decoder_inputs)  # Outputs length same as decoder inputs.
        state_list = outputs_and_state[outputs_len:]
        state = state_list[0]
        if nest.is_sequence(encoder_state):
            state = nest.pack_sequence_as(structure=encoder_state,
                                          flat_sequence=state_list)
        return outputs_and_state[:outputs_len], state


def sequence_loss_by_example(logits, targets, weights,
                             average_across_timesteps=True,
                             softmax_loss_function=None, name=None):
    #print("Inside the method sequence loss by example")

    if len(targets) != len(logits) or len(weights) != len(logits):
        raise ValueError("Lengths of logits, weights, and targets must be the same "
                         "%d, %d, %d." % (len(logits), len(weights), len(targets)))
    with ops.name_scope(name, "sequence_loss_by_example",
                        logits + targets + weights):
        log_perp_list = []
        for logit, target, weight in zip(logits, targets, weights):
            if softmax_loss_function is None:
                # TODO(irving,ebrevdo): This reshape is needed because
                # sequence_loss_by_example is called with scalars sometimes, which
                # violates our general scalar strictness policy.
                target = array_ops.reshape(target, [-1])
                crossent = nn_ops.sparse_softmax_cross_entropy_with_logits(
                        logit, target)
            else:
                crossent = softmax_loss_function(logit, target)
            log_perp_list.append(crossent * weight)
        log_perps = math_ops.add_n(log_perp_list)
        if average_across_timesteps:
            total_size = math_ops.add_n(weights)
            total_size += 1e-12  # Just to avoid division by 0 for all-0 weights.
            log_perps /= total_size
    return log_perps


def sequence_loss(logits, targets, weights,
                  average_across_timesteps=True, average_across_batch=True,
                  softmax_loss_function=None, name=None):
    #print("Inside the method sequence loss")
    with ops.name_scope(name, "sequence_loss", logits + targets + weights):
        cost = math_ops.reduce_sum(sequence_loss_by_example(
                logits, targets, weights,
                average_across_timesteps=average_across_timesteps,
                softmax_loss_function=softmax_loss_function))
        if average_across_batch:
            batch_size = array_ops.shape(targets[0])[0]
            return cost / math_ops.cast(batch_size, cost.dtype)
        else:
            return cost


def model_with_buckets(encoder_inputs, context_inputs, decoder_inputs, targets, weights,
                       buckets, seq2seq, softmax_loss_function=None,
                       per_example_loss=False, name=None):
    """Create a sequence-to-sequence model with support for bucketing.

    The seq2seq argument is a function that defines a sequence-to-sequence model,
    e.g., seq2seq = lambda x, y: basic_rnn_seq2seq(x, y, rnn_cell.GRUCell(24))

    Args:
      encoder_inputs: A list of Tensors to feed the encoder; first seq2seq input.
      decoder_inputs: A list of Tensors to feed the decoder; second seq2seq input.
      targets: A list of 1D batch-sized int32 Tensors (desired output sequence).
      weights: List of 1D batch-sized float-Tensors to weight the targets.
      buckets: A list of pairs of (input size, output size) for each bucket.
      seq2seq: A sequence-to-sequence model function; it takes 2 input that
        agree with encoder_inputs and decoder_inputs, and returns a pair
        consisting of outputs and states (as, e.g., basic_rnn_seq2seq).
      softmax_loss_function: Function (inputs-batch, labels-batch) -> loss-batch
        to be used instead of the standard softmax (the default if this is None).
      per_example_loss: Boolean. If set, the returned loss will be a batch-sized
        tensor of losses for each sequence in the batch. If unset, it will be
        a scalar with the averaged loss from all examples.
      name: Optional name for this operation, defaults to "model_with_buckets".

    Returns:
      A tuple of the form (outputs, losses), where:
        outputs: The outputs for each bucket. Its j'th element consists of a list
          of 2D Tensors. The shape of output tensors can be either
          [batch_size x output_size] or [batch_size x num_decoder_symbols]
          depending on the seq2seq model used.
        losses: List of scalar Tensors, representing losses for each bucket, or,
          if per_example_loss is set, a list of 1D batch-sized float Tensors.

    Raises:
      ValueError: If length of encoder_inputsut, targets, or weights is smaller
        than the largest (last) bucket.
    """
    if len(encoder_inputs) < buckets[-1][0]:
        raise ValueError("Length of encoder_inputs (%d) must be at least that of la"
                         "st bucket (%d)." % (len(encoder_inputs), buckets[-1][0]))
    if len(targets) < buckets[-1][1]:
        raise ValueError("Length of targets (%d) must be at least that of last"
                         "bucket (%d)." % (len(targets), buckets[-1][1]))
    if len(weights) < buckets[-1][1]:
        raise ValueError("Length of weights (%d) must be at least that of last"
                         "bucket (%d)." % (len(weights), buckets[-1][1]))

    #print("In method model with buckets")

    print("The encoder input shape {0}".format(np.shape(encoder_inputs)))
    print("The decoder input shape {0}".format(np.shape(decoder_inputs)))
    print("The targets input shape {0}".format(np.shape(targets)))
    print("The weights input shape {0}".format(np.shape(weights)))

    all_inputs = encoder_inputs + decoder_inputs + targets + weights
    losses = []
    outputs = []
    #print("Calling the function embedding_attention_seq2seq {0} times".format(len(buckets)))
    with ops.name_scope(name, "model_with_buckets", all_inputs):
        for j, bucket in enumerate(buckets):
            with variable_scope.variable_scope(variable_scope.get_variable_scope(),
                                               reuse=True if j > 0 else None):

                bucket_outputs, _ = seq2seq(encoder_inputs[:bucket[0]],
                                            context_inputs[:bucket[0]],
                                            decoder_inputs[:bucket[1]])
                outputs.append(bucket_outputs)
                if per_example_loss:
                    losses.append(sequence_loss_by_example(
                            outputs[-1], targets[:bucket[1]], weights[:bucket[1]],
                            softmax_loss_function=softmax_loss_function))
                else:
                    losses.append(sequence_loss(
                            outputs[-1], targets[:bucket[1]], weights[:bucket[1]],
                            softmax_loss_function=softmax_loss_function))

    return outputs, losses
