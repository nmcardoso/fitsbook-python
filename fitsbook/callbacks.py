from keras.callbacks import Callback
from keras import optimizers

import requests
import numpy as np
import json
import os

class FitsbookCallback(Callback):
  def __init__(self):
    if os.environ.get('PY_ENV', '') == 'DEV':
      self.api_root = 'http://localhost:3000/api'
    else:
      self.api_root = 'https://fitsbook.glitch.me/api'
      
    self.model_id = None

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
      'model': {
        'name': self.model.name or '',
        'config': self.model.get_config()
      },
      'optimizer': {
        'name': opt_name,
        'config': self.model.optimizer.get_config()
      }
    }

    response = requests.post(f'{self.api_root}/model', json=send)
    print(f'{self.api_root}')
    if (response):
      r = response.json()
      self.model_id = r['id'] if r['id'] else None

  def on_epoch_end(self, epoch, logs=None):
    _logs = {}
    for k, v in logs.items():
      if isinstance(v, (np.ndarray, np.generic)):
        _logs[k] = v.item()
      else:
        _logs[k] = v
    
    send = {
      'epoch': epoch,
      'metrics': _logs
    }

    requests.post(f'{self.api_root}/history/{self.model_id}', json=send)