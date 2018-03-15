"""Construct a FaceID model"""
import tensorflow as tf

_INIT_WEIGHTS = True

def resnet_carid(multi_gpu):
  func = resnet50_model_fn if not multi_gpu else \
      tf.contrib.estimator.replicate_model_fn(
         resnet50_model_fn, tf.losses.Reduction.MEAN)
  return func


def resnet50_model_fn(features, labels, mode, params):
  """Model Function for tf.estimator.Estimator object

  Note that because of triplet loss function, we do not need labels
  """
  global _INIT_WEIGHTS

  # Determine if model should update weights
  tf.keras.backend.set_learning_phase(mode == tf.estimator.ModeKeys.TRAIN)
  model = tf.keras.applications.ResNet50(
      input_tensor=tf.keras.Input(tensor=features),
      include_top=False,
      pooling='avg',
      weights='imagenet' if _INIT_WEIGHTS else None)

  if _INIT_WEIGHTS:
    print('Imagenet weights have been loaded into Resnet.')
    _INIT_WEIGHTS = False  # only init one time

  outputs = model(features)

  # @TODO: batch-hard triplet loss
  loss = tf.reduce_mean(outputs)

  if mode == tf.estimator.ModeKeys.TRAIN:
    global_step = tf.train.get_or_create_global_step()
    optimizer = params['optimizer'](params['learning_rate'])

    if params['multi_gpu']:
      optimizer = tf.contrib.estimator.TowerOptimizer(optimizer)

    # for batch_norm
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
      train_ops = optimizer.minimize(loss, global_step)
  else:
    train_ops = None

  # Add Summary
  tf.identity(loss, 'train_loss')
  tf.summary.scalar('train_loss', loss)

  predictions = {
    'out': outputs
  }
  return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions,
      loss=loss,
      train_op=train_ops,
      eval_metric_ops={})

