import math
import numpy as np

#make pattern
def make_cos_pattern(patterns):
  ts = np.arange(0.0, 32, 1.0)
  cols = [np.arange(0) for _ in range(16)]
  for p in patterns:
    tp, sp = p
    a = -1.0
    for i in range(16):
      if i % sp == 0:
        a *= -1.0
      data = a*np.cos(2.0*math.pi * tp * ts)
      cols[i] = np.concatenate([cols[i],data])
  return cols
  
def dataset():
  spatial_patterns = [2, 4, 8]
  temporal_patterns = []
  for s in spatial_patterns:
    temporal_patterns.append(1.0/s)
  all_pattern = []
  for sp in spatial_patterns:
    for tp in temporal_patterns:
      #input set_pattern ex 2,1/2
      all_pattern.append((tp, sp))
  inputs = make_cos_pattern(all_pattern)
  #サイズの自動調整機能.view!
  inputs = np.reshape(np.stack(inputs).T, [9, 32, 16])

  train_x = [inputs]
  train_y = [[0, 0, 0, 1, 1, 1, 2, 2, 2],[0, 1, 2, 0, 1, 2, 0, 1, 2]]

  train_loader = [(x,y) for x, y in zip(train_x, train_y)]

  test_x = inputs
  test_y = [[0, 0, 0, 1, 1, 1, 2, 2, 2],[0, 1, 2, 0, 1, 2, 0, 1, 2]]
  
  return train_loader, test_x, test_y
