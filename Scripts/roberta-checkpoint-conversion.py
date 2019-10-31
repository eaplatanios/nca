import numpy as np
import tensorflow as tf
import torch
import os

variables = torch.load('model.pt', map_location='cpu')['model']

def create_tf_variable(value, name, session):
  print('Loading variable ' + name + ' with data type ' + str(value.dtype) + '.')
  tf_variable = tf.get_variable(
    dtype=tf.float32,
    shape=value.shape,
    name=name,
    initializer=tf.zeros_initializer())
  session.run(tf.variables_initializer([tf_variable]))
  session.run(tf_variable)
  tf.keras.backend.set_value(tf_variable, value.astype(np.float32))
  session.run(tf_variable)

tf.reset_default_graph()
with tf.Session() as session:
  create_tf_variable(
    value=variables['decoder.sentence_encoder.embed_tokens.weight'].numpy(),
    name='bert/embeddings/word_embeddings',
    session=session)
  create_tf_variable(
    value=variables['decoder.sentence_encoder.embed_positions.weight'].numpy(),
    name='bert/embeddings/position_embeddings',
    session=session)
  create_tf_variable(
    value=variables['decoder.sentence_encoder.emb_layer_norm.weight'].numpy(),
    name='bert/embeddings/LayerNorm/beta',
    session=session)
  create_tf_variable(
    value=variables['decoder.sentence_encoder.emb_layer_norm.bias'].numpy(),
    name='bert/embeddings/LayerNorm/gamma',
    session=session)
  for layer in range((len(variables) - 10) // 12):
    prefix = 'decoder.sentence_encoder.layers.' + str(layer)
    tf_prefix = 'bert/encoder/layer_' + str(layer)
    attention_kernels = variables[prefix + '.self_attn.in_proj_weight'].numpy()
    attention_biases = variables[prefix + '.self_attn.in_proj_bias'].numpy()
    hidden_size = attention_kernels.shape[0] // 3
    create_tf_variable(
      value=np.transpose(attention_kernels[:hidden_size, :]),
      name=tf_prefix + '/attention/self/query/kernel',
      session=session)
    create_tf_variable(
      value=attention_biases[:hidden_size],
      name=tf_prefix + '/attention/self/query/bias',
      session=session)
    create_tf_variable(
      value=np.transpose(attention_kernels[hidden_size:2*hidden_size, :]),
      name=tf_prefix + '/attention/self/key/kernel',
      session=session)
    create_tf_variable(
      value=attention_biases[hidden_size:2*hidden_size],
      name=tf_prefix + '/attention/self/key/bias',
      session=session)
    create_tf_variable(
      value=np.transpose(attention_kernels[2*hidden_size:, :]),
      name=tf_prefix + '/attention/self/value/kernel',
      session=session)
    create_tf_variable(
      value=attention_biases[2*hidden_size:],
      name=tf_prefix + '/attention/self/value/bias',
      session=session)
    create_tf_variable(
      value=np.transpose(variables[prefix + '.self_attn.out_proj.weight'].numpy()),
      name=tf_prefix + '/attention/output/dense/kernel',
      session=session)
    create_tf_variable(
      value=variables[prefix + '.self_attn.out_proj.bias'].numpy(),
      name=tf_prefix + '/attention/output/dense/bias',
      session=session)
    create_tf_variable(
      value=variables[prefix + '.self_attn_layer_norm.weight'].numpy(),
      name=tf_prefix + '/attention/output/LayerNorm/beta',
      session=session)
    create_tf_variable(
      value=variables[prefix + '.self_attn_layer_norm.bias'].numpy(),
      name=tf_prefix + '/attention/output/LayerNorm/gamma',
      session=session)
    create_tf_variable(
      value=np.transpose(variables[prefix + '.fc1.weight'].numpy()),
      name=tf_prefix + '/intermediate/dense/kernel',
      session=session)
    create_tf_variable(
      value=variables[prefix + '.fc1.bias'].numpy(),
      name=tf_prefix + '/intermediate/dense/bias',
      session=session)
    create_tf_variable(
      value=np.transpose(variables[prefix + '.fc2.weight'].numpy()),
      name=tf_prefix + '/output/dense/kernel',
      session=session)
    create_tf_variable(
      value=variables[prefix + '.fc2.bias'].numpy(),
      name=tf_prefix + '/output/dense/bias',
      session=session)
    create_tf_variable(
      value=variables[prefix + '.final_layer_norm.weight'].numpy(),
      name=tf_prefix + '/output/LayerNorm/beta',
      session=session)
    create_tf_variable(
      value=variables[prefix + '.final_layer_norm.bias'].numpy(),
      name=tf_prefix + '/output/LayerNorm/gamma',
      session=session)
  saver = tf.train.Saver(tf.trainable_variables())
  saver.save(session, os.path.join(os.getcwd(), 'roberta_large.ckpt'))
