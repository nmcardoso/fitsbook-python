import numpy as np

from keras import layers
from keras import models

def main():
  # Simple model to test API
  model = models.Sequential()
  model.add(layers.Dense(64, activation='relu', input_shape=(1)))
  model.add(layers.Dense(64, activation='relu'))
  model.add(layers.Dense(1))

  model.compile(
    loss='mean_squared_error', 
    optimizer='rmsprop', 
    metrics=['mean_absolute_error', 'mean_squared_error']
  )

  fahrenheit=np.array([-140,-136,-124,-112,-105,-96,-88,-75,-63,-60,-58,-40,-20,-10,0,30,35,48,55,69,81,89,95,99,105,110,120,135,145,158,160],dtype=float)
  celsius=np.array([-95.55,-93.33,-86.66,-80,-76.11,-71.11,-66.66,-59.44,-52.77,-51.11,-50,-40,-28.88,-23.33,-17.77,-1.11,1.66,8.88,12,20,27.22,31.66,35,37.22,40.55,43.33,48.88,57.22,62.77,70,71.11],dtype=float)

  model.fit(Fahrenheit,Celsius,epochs=10)

if __name__ == '__main__':
  main()