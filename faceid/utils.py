"""Visualize Embeddings on Tensorboard using Projector plug-in."""
import os
import numpy as np
from scipy.misc import imsave

import tensorflow as tf
from tensorboard.plugins import projector

def visualize_embeddings(images, embeddings, output_dir,
                         thumbnail_size=(32, 32)):
  """

  Args:
    images:
    embeddings:
    output_dir:
    thumbnail_size:

  Returns:

  """
  summary_writer = tf.summary.FileWriter(output_dir)

  sprite_path = os.path.abspath(os.path.join(output_dir, 'sprite.png'))
  metadata_path = os.path.abspath(os.path.join(output_dir, 'metadata.csv'))
  embeddings_path = os.path.join(output_dir, 'embeddings.ckpt')

  embedding_var = tf.Variable(embeddings, name='embeddings')
  sprite = images_to_sprite(images)
  imsave(os.path.join(output_dir, 'sprite.png'), sprite)

  with tf.Session() as sess:
    sess.run(embedding_var.initializer)
    config = projector.ProjectorConfig()

    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name
    embedding.metadata_path = metadata_path
    embedding.sprite.image_path = sprite_path
    embedding.sprite.single_image_dim.extend(thumbnail_size)

    projector.visualize_embeddings(summary_writer, config)
    saver = tf.train.Saver([embedding_var])
    saver.save(sess, embeddings_path, 1)

def images_to_sprite(data):
  """Creates the sprite image along with any necessary padding.
  Taken from: https://github.com/tensorflow/tensorflow/issues/6322
  Args:
    data: NxHxW[x3] tensor containing the images.
  Returns:
    data: Properly shaped HxWx3 image with any necessary padding.
  """
  if len(data.shape) == 3:
    data = np.tile(data[..., np.newaxis], (1, 1, 1, 3))
  data = data.astype(np.uint8)
  n = int(np.ceil(np.sqrt(data.shape[0])))
  padding = ((0, n ** 2 - data.shape[0]), (0, 0),
             (0, 0)) + ((0, 0),) * (data.ndim - 3)
  data = np.pad(data, padding, mode='constant',
                constant_values=0)
  # Tile the individual thumbnails into an image.
  data = data.reshape((n, n) + data.shape[1:]).transpose(
      (0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
  data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
  return data
