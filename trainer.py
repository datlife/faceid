"""Train FaceID model"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import argparse

import faceid
import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
_RGB_MEAN = [123.68, 116.78, 103.94]


#######################################
# Data processing
#######################################
def _read_py_function(filename, mode):
  def cv2_read(img_path):   # this runs x2 faster than tf.read_file
    image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    image = cv2.resize(image,  (224, 224)).astype(np.float32)
    image = preprocess_fn(image, mode)
    return image
  image = tf.py_func(cv2_read, [filename], tf.float32)
  image.set_shape([224, 224, 3])
  return image

# @TODO: data augmentation
def preprocess_fn(image, mode):
  image -= np.expand_dims(np.expand_dims(_RGB_MEAN, 0), 0)
  image /= 255.0
  return image


def main():
  # ##########################
  # Configure hyper-parameters
  # ##########################
  args = parse_args()
  batch_size = args.batch_size
  training_steps = args.steps
  steps_per_epoch = args.steps_per_epoch
  epochs_per_eval = 3
  training_epochs = int(training_steps // steps_per_epoch)

  cpu_cores = 8
  multi_gpu = True
  shuffle_buffer = batch_size * cpu_cores
  # ########################
  # Load VeRi dataset
  # ########################
  veri_dataset = faceid.dataset.VeRiDataset(root_dir=args.dataset_dir).load()

  # ########################
  # Define a Classifier
  # ########################
  estimator = tf.estimator.Estimator(
      model_fn=faceid.resnet_carid(multi_gpu=multi_gpu),
      model_dir=args.model_dir,
      config=tf.estimator.RunConfig().replace(
          save_checkpoints_steps=steps_per_epoch,
          save_summary_steps=steps_per_epoch,
          log_step_count_steps=steps_per_epoch),
      params={
          'learning_rate': 0.001,
          'weight_decay': 2e-4,
          'optimizer': tf.train.AdamOptimizer,
          'multi_gpu': multi_gpu,
          'loss_function': faceid.triplet_loss,
          'margin': 0.2
      })

  # #########################
  # Training/Eval
  # #########################
  tensors_to_log = ['train_loss']
  for _ in range(training_epochs // epochs_per_eval):
    train_data, eval_data = veri_dataset.split_training_data(
        test_size=0.2,
        shuffle=True)

    estimator.train(
        input_fn=lambda: veri_dataset.get_input_fn(
            mode=tf.estimator.ModeKeys.TRAIN,
            dataset=train_data,  # pylint: disable=cell-var-from-loop
            batch_size=batch_size,
            parse_fn=_read_py_function,
            shuffle_buffer=shuffle_buffer,
            num_parallel_calls=cpu_cores),
        steps=steps_per_epoch * epochs_per_eval,
        hooks=[faceid.ProgressBarHook(
            epochs=int(training_steps // steps_per_epoch),
            steps_per_epoch=steps_per_epoch,
            tensors_to_log=tensors_to_log)])

    print("\nStart evaluating...")
    eval_result = estimator.evaluate(
        input_fn=lambda: veri_dataset.get_input_fn(
            mode=tf.estimator.ModeKeys.EVAL,
            dataset=eval_data,  # pylint: disable=cell-var-from-loop
            batch_size=batch_size,
            parse_fn=_read_py_function,
            shuffle_buffer=None,
            num_parallel_calls=cpu_cores),
        steps=200)
    print(eval_result)
  print("---- Training Completed ----")


def parse_args():
  """Command line arguments"""
  parser = argparse.ArgumentParser(
      description="Train a Car Re-identification model")
  parser.add_argument(
      "--dataset_dir", help="Path to dataset directory.",
      default=None, required=True)
  parser.add_argument(
      "--batch_size", help="Number of training instances for every iteration",
      default=48, type=int)
  parser.add_argument(
      "--steps", help="Number of iteration per epochs",
      default=30e3, type=int)
  parser.add_argument(
      "--steps_per_epoch", help="Number of iteration per epochs",
      default=500, type=int)
  parser.add_argument(
      "--model_dir", help="Path to store training log and trained model",
      default=None)
  return parser.parse_args()


if __name__ == '__main__':
  main()
