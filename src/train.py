import input_data
import model

import math
import numpy as np

def ridge_ym(model, train_loader, test_x, test_y):
  print(model)
  epochs = []
  accuracy_sp = []
  accuracy_tp = []
  loss_tp = []
  loss_sp = []

  return epochs, accuracy_sp, accuracy_tp, loss_sp, loss_tp

  


if __name__ == '__main__':
  train_loader, test_x, test_y = input_data.dataset() 

  train_x, train_y = train_loader[0]

  print(train_x)

  epochs, accuracy_sp, accuracy_tp,loss_sp, loss_tp= ridge_ym(model.BindeESN(size_in=16,size_res=32,size_out=3,leaky=0.3), train_loader, test_x, test_y)

