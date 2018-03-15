r"""Dataset loader for Triplet Model

"""

import abc
from sklearn.model_selection import train_test_split


class DataProvider(object):
  """"Abstract base class for Dataset object
  """
  __metaclass__ = abc.ABCMeta

  def __init__(self, root_dir):
    """Instantiate `DataProvider` instance

    Args:
      root_dir: - String - absolute path
        to dataset directory. It should contain all necessary training/testing
        instances and labels.

    """
    self.root_dir = root_dir
    self.data = None

  @abc.abstractmethod
  def load(self):
    """"Initialize `self.data` for other methods to use"""
    pass

  @abc.abstractmethod
  def get_input_fn(self,
                   mode,
                   data,
                   batch_size,
                   parse_record_fn,
                   shuffle_buffer=200,
                   num_parallel_calls=4):
    """Create a input function"""
    pass

  def split_training_data(self, test_size, shuffle=False):
    if self.data is None:
      raise ValueError("Data is currently empty. Did you call load()?")
    training_data, validation_data = train_test_split(
        self.data,
        test_size=test_size,
        shuffle=shuffle)
    return training_data, validation_data


