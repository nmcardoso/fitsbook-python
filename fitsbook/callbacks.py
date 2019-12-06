from keras.callbacks import Callback

import requests
import numpy as np

class TestCallback(Callback):
  def on_train_begin(self, logs=None):
    logs = logs or {}
    _logs = {}
    for k, v in logs.items():
      if isinstance(v, (np.ndarray, np.generic)):
        _logs[k] = v.item()
      else:
        _logs[k] = v

    send = {
      'event': 'on_train_begin',
      'model': self.model,
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