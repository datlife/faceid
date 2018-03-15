"""Custom Training Hooks"""
import numpy as np

from tensorflow.python.training import training_util
from tensorflow.python.training import session_run_hook

from tensorflow.python.training.session_run_hook import SessionRunArgs
from tensorflow.python.training.basic_session_run_hooks import _as_graph_element
from tensorflow.python.keras.utils import Progbar

# @tf_export('train.ProgressBarHook')
class ProgressBarHook(session_run_hook.SessionRunHook):
  """Monitors training progress. This hook uses `tf.keras.utils.ProgBar` to
  write messages to stdout.

  Example:
    ```python
    estimator = tf.estimator.DNNClassifier(hidden_units=256, feature_columns=64)
    estimator.train(
      input_fn=lambda: input_fn,
      hooks=[ProgressBar(
        epochs=3,
        steps_per_epoch=4,
        tensors_to_log=['loss', 'acc'])])
    ```
    # output
    ```
    Epoch 1/5:
    4/4 [======================]4/4 - 13s 3s/step - acc: 0.7 - loss: 0.4124

    Epoch 2/5:
    4/4 [======================]4/4 - 1s 175ms/step - acc: 0.7235 - loss: 0.2313

    Epoch 3/5:
    4/4 [======================]4/4 - 1s 168ms/step - acc: 0.7814 - loss: 0.1951
    ```

  """
  def __init__(self,
               epochs,
               steps_per_epoch,
               tensors_to_log=None):
    """Initializes `ProgressBarHook` instance

    Args:
      epochs: `int`, Total number of expected epochs. It is usually calcuated
        by dividing number of training steps to `steps_per_epoch`.
      steps_per_epoch: `int`, numbers of expected iterations per epoch
      tensors_to_log: - optional - can be:
          `dict` maps string-valued tags to tensors/tensor names,
          or `iterable` of tensors/tensor names.

    Raise:
      ValueError: `tensors_to_log` is not a list or a dictionary.
    """
    self._epochs = epochs
    self._step_per_epoch = steps_per_epoch

    if tensors_to_log is not None:
      if not isinstance(tensors_to_log, dict):
        self._tag_order = tensors_to_log
        tensors_to_log = {item: item for item in tensors_to_log}
      else:
        self._tag_order = tensors_to_log.keys()
      self._tensors = tensors_to_log
    else:
      self._tensors = None

  def begin(self):
    self._global_step_tensor = training_util._get_global_step_read()  # pylint: disable=protected-access
    if self._global_step_tensor is None:
      raise RuntimeError(
          "Global step should be created to use ProgressBarHook")

    # Convert names to tensors if given
    if self._tensors:
      self._current_tensors = {tag: _as_graph_element(tensor)
                               for (tag, tensor) in self._tensors.items()}

  def after_create_session(self, session, coord):  # pylint: disable=unused-argument
    # Init current_epoch and current_step
    self._curr_step = session.run(self._global_step_tensor)
    if self._curr_step != 0:
      print('Resuming training from global step(s): %s...\n' % self._curr_step)

    self._curr_epoch = int(np.floor(self._curr_step / self._step_per_epoch))
    self._curr_step -= self._curr_epoch * self._step_per_epoch
    self._first_run = True

  def before_run(self, run_context):  # pylint: disable=unused-argument
    if self._first_run is  True:
      self._curr_epoch += 1
      print('Epoch %s/%s:' % (self._curr_epoch, self._epochs))
      self.progbar = Progbar(target=self._step_per_epoch)
      self._first_run = False

    elif self._curr_step % self._step_per_epoch == 0:
      self._curr_epoch += 1
      self._curr_step = 0
      print('Epoch %s/%s:' % (self._curr_epoch, self._epochs))
      self.progbar = Progbar(target=self._step_per_epoch)

    if self._tensors:
      return SessionRunArgs(self._current_tensors)

    return None

  def after_run(self,
                run_context,  # pylint: disable=unused-argument
                run_values):
    if self._tensors:
      values = self._extract_tensors_info(run_values.results)
    else:
      values = None
    self._curr_step += 1
    self.progbar.update(self._curr_step, values=values)

  def _extract_tensors_info(self, tensor_values):
    stats = []
    for tag in self._tag_order:
      stats.append((tag, tensor_values[tag]))
    return stats
