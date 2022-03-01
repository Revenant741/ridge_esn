import csv
import pickle
import math
import numpy as np
import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt
from input import input_np
from input import inputdata
import sklearn.metrics
import cloudpickle

class Analysis:
  def __init__(self,name):
    self.name = name
    
  def save_to_data(self, acc_data, loss_data):
    sp_accuracy, tp_accuracy = acc_data
    print(sp_accuracy)
    print(type(sp_accuracy))
    sp_loss, tp_loss = loss_data
    #point = str(os.path.dirname(os.path.abspath(__file__)))+'/data/'
    point = 'data/'
    name = self.name
    #torch.save(model.to('cpu').state_dict(), point+name+'_model.pth')
    with open(f''+point+name+'_sp_acc.csv', 'w') as f:
        writer = csv.writer(f)
        for accuracy1 in sp_accuracy:
            writer.writerow([accuracy1])
    with open(f''+point+name+'_tp_acc.csv', 'w') as f:
        writer = csv.writer(f)
        for accuracy2 in tp_accuracy:
            writer.writerow([accuracy2])
    with open(f''+point+name+'_sp_loss.csv', 'w') as f:
        writer = csv.writer(f)
        for loss1 in sp_loss:
            writer.writerow([loss1])
    with open(f''+point+name+'_tp_loss.csv', 'w') as f:
        writer = csv.writer(f)
        for loss2 in tp_loss:
            writer.writerow([loss2])
  
  def save_to_weight(self, weight_res, b_res, sp_weight_out, tp_weight_out):
    point = 'data/'
    name = self.name
    with open(f''+point+name+'_weight_res.dat', 'wb') as f:
      for W_res in weight_res:
        cloudpickle.dump(W_res, f)
    with open(f''+point+name+'_b_res.dat', 'wb') as f:
      for B_res in b_res:
        cloudpickle.dump(B_res, f)
    with open(f''+point+name+'_sp_weight_out.dat', 'wb') as f:
      for W_spout in sp_weight_out:
        cloudpickle.dump(W_spout, f)
    with open(f''+point+name+'_tp_weight_out.dat', 'wb') as f:
      for W_tpout in tp_weight_out:
        cloudpickle.dump(W_tpout, f)

  def save_to_mutual(self, h_in_x, h_in_y, h_out_x, h_out_y):
    name = self.name
    #point = str(os.path.dirname(os.path.abspath(__file__)))+'/data/'
    point = 'data/'
    with open(f''+point+name+'_h_in_x.csv', 'w') as f:
        writer = csv.writer(f)
        for x_1 in h_in_x:
            writer.writerow([x_1])
    with open(f''+point+name+'_h_in_y.csv', 'w') as f:
      writer = csv.writer(f)
      for y_1 in h_in_y:
          writer.writerow([y_1])
    with open(f''+point+name+'_h_out_x.csv', 'w') as f:
        writer = csv.writer(f)
        for x_2 in h_out_x:
            writer.writerow([x_2])
    with open(f''+point+name+'_h_out_y.csv', 'w') as f:
      writer = csv.writer(f)
      for y_2 in h_out_y:
          writer.writerow([y_2])

  def save_in_layer_para(self,weight_in, b_in, b_res):
    name = self.name
    point = 'ga_data/'
    with open(f''+point+name+'_bias_weight_in.dat', 'wb') as f:
      pickle.dump(weight_in, f)
    with open(f''+point+name+'_bias_in.dat', 'wb') as f:
      pickle.dump(b_in, f)
    with open(f''+point+name+'_bias_res.dat', 'wb') as f:
      pickle.dump(b_res, f)
  
  def save_to_ga_data(self,SP_A,TP_A,SP_L,TP_L,G,W_res,B_res,SP_W,TP_W):
    name = self.name
    point = 'ga_data/'
    with open(f''+point+name+'_sp_acc.csv', 'w') as f:
        writer = csv.writer(f)
        for sp_accs in SP_A:
          writer.writerow([sp_accs])
    with open(f''+point+name+'_tp_acc.csv', 'w') as f:
        writer = csv.writer(f)
        for tp_accs in TP_A:
          writer.writerow([tp_accs])
    with open(f''+point+name+'_sp_loss.csv', 'w') as f:
        writer = csv.writer(f)
        for sp_losses in SP_L:
          writer.writerow([sp_losses])
    with open(f''+point+name+'_tp_loss.csv', 'w') as f:
          writer = csv.writer(f)
          for tp_losses in TP_L:
            writer.writerow([tp_losses])
    # generation = G[-1]
    # if generation%100 == 0:
    with open(f''+point+name+'_weight_res.dat', 'wb') as f:
      pickle.dump(W_res, f)
    with open(f''+point+name+'_bias_res.dat', 'wb') as f:
      pickle.dump(B_res, f)
    with open(f''+point+name+'_weight_sp_out.dat', 'wb') as f:
      pickle.dump(SP_W, f)
    with open(f''+point+name+'_weight_tp_out.dat', 'wb') as f:
      pickle.dump(TP_W, f)
    '''
    with open(f'ga_data/generation.csv', 'w') as f:
        writer = csv.writer(f)
        for generation in G:
          writer.writerow([generation])
    '''

  def save_to_ga_binde_data(self,SP_A,TP_A,SP_L,TP_L,G,W_res1,W_res2,W_res3,W_res4,SP_W,TP_W):
    name = self.name
    point = 'ga_data/'
    with open(f''+point+name+'_sp_acc.csv', 'w') as f:
        writer = csv.writer(f)
        for sp_accs in SP_A:
          writer.writerow([sp_accs])
    with open(f''+point+name+'_tp_acc.csv', 'w') as f:
        writer = csv.writer(f)
        for tp_accs in TP_A:
          writer.writerow([tp_accs])
    with open(f''+point+name+'_sp_loss.csv', 'w') as f:
        writer = csv.writer(f)
        for sp_losses in SP_L:
          writer.writerow([sp_losses])
    with open(f''+point+name+'_tp_loss.csv', 'w') as f:
          writer = csv.writer(f)
          for tp_losses in TP_L:
            writer.writerow([tp_losses])
    # generation = G[-1]
    # if generation%100 == 0:
    with open(f''+point+name+'_weight_res1.dat', 'wb') as f:
      pickle.dump(W_res1, f)
    with open(f''+point+name+'_weight_res2.dat', 'wb') as f:
      pickle.dump(W_res2, f)
    with open(f''+point+name+'_weight_res3.dat', 'wb') as f:
      pickle.dump(W_res3, f)
    with open(f''+point+name+'_weight_res4.dat', 'wb') as f:
      pickle.dump(W_res4, f)
    with open(f''+point+name+'_weight_sp_out.dat', 'wb') as f:
      pickle.dump(SP_W, f)
    with open(f''+point+name+'_weight_tp_out.dat', 'wb') as f:
      pickle.dump(TP_W, f)
    with open(f''+point+name+'generation.csv', 'w') as f:
        writer = csv.writer(f)
        for generation in G:
          writer.writerow([generation])

  def save_to_ga_mutual_data(self, h_in_x, h_in_y, h_out_x, h_out_y):
    name = self.name
    #point = str(os.path.dirname(os.path.abspath(__file__)))+'/data/'
    point = 'ga_data/'
    with open(f''+point+name+'_h_in_x.csv', 'w') as f:
        writer = csv.writer(f)
        for x_1 in h_in_x:
            writer.writerow([x_1])
    with open(f''+point+name+'_h_in_y.csv', 'w') as f:
      writer = csv.writer(f)
      for y_1 in h_in_y:
          writer.writerow([y_1])
    with open(f''+point+name+'_h_out_x.csv', 'w') as f:
        writer = csv.writer(f)
        for x_2 in h_out_x:
            writer.writerow([x_2])
    with open(f''+point+name+'_h_out_y.csv', 'w') as f:
      writer = csv.writer(f)
      for y_2 in h_out_y:
          writer.writerow([y_2])