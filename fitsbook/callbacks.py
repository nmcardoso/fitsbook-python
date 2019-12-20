from keras.callbacks import Callback
from keras import optimizers

import requests
import numpy as np
import json
import os

class FitsbookCallback(Callback):
  def __init__(self):
    if os.environ.get('PY_ENV', '') == 'DEV':
      self.api_root = 'http://localhost:8000/api'
      self.site_url = 'http://localhost:3000/#'
    else:
      self.api_root = 'https://fitsbook.glitch.me/api'
      self.site_url = 'https://natan.ninja/#'
      
    self.model_id = None
    self.remote_stop = False # internal flag

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
    if (response):
      r = response.json()
      self.model_id = r['id'] if r['id'] else None
      print(f'[Fitsbook]: Monitoring this training in real time {self.site_url}/stats/{self.model_id}')
  
  def on_train_end(self, logs=None):
    response = requests.post(f'{self.api_root}/training/{self.model_id}/end')
    if response and response.status_code == 200:
      print(f'[Fitsbook]: Training ended successfuly. {self.site_url}/stats/{self.model_id}')
    else:
      print('[Fitsbook]: Some error occurred while trying to finish training.')

    if (self.remote_stop):
      print('[Fitsbook]: Remotely ended training via webapp.')

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

    response = requests.get(f'{self.api_root}/training/{self.model_id}/stop')
    if (response):
      r = response.json()
      if (r['stop']):
        self.model.stop_training = True
        self.remote_stop = True