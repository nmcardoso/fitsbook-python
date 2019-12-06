from keras.callbacks import Callback
from keras import optimizers

import requests
import numpy as np
import json

class FitsbookCallback(Callback):
  def on_train_begin(self, logs=None):
    logs = logs or {}
    _logs = {}
    for k, v in logs.items():
      if isinstance(v, (np.ndarray, np.generic)):
        _logs[k] = v.item()
      else:
        _logs[k] = v

    if type(self.model.optimizer) is str:
      opt_name = optimizers.get(self.model.optimizers).__class__.__name__
    else:
      opt_name = self.model.optimizer.__class__.__name__
  
    send = {
      'event': 'on_train_begin',
      'model_name': self.model.name or '',
      'model_config': self.model.get_config(),
      'optimizer_name': opt_name,
      'optimizer_config': self.model.optimizer.get_config(),
      'log': _logs
    }

    requests.post('http://fitsbook.glitch.me/test', json=send)

  def on_epoch_end(self, epoch, logs=None):
    _logs = {}
    for k, v in logs.items():
      if isinstance(v, (np.ndarray, np.generic)):
        _logs[k] = v.item()
      else:
        _logs[k] = v
    
    send = {
      'event': 'on_epoch_end',
      'epoch': epoch,
      'log': _logs
    }

    requests.post('http://fitsbook.glitch.me/test', json=send)