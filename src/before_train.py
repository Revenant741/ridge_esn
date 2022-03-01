import math
import numpy as np
import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt
from input import input_np
from input import inputdata
from model import esn_model as ESN_Model
from model import leaky_esn_model as LeakyESN_Model
from my_def import Analysis
import os
import csv
import cloudpickle
import sklearn.metrics

class Ridge_train:
  def __init__(self, model, inputdata):
    self.inputdata = inputdata
    self.model = model
  #main関数
  def main(self, weight_in, weight_res_in,weight_res_out, b_in, b_res):
    #入力データを取得
    input, sp_ans, tp_ans = self.inputdata()
    model = self.model
    #input, sp_ans, tp_ans = inputdata.makeinput()
    #パラメータの設定
    sp_acc = 0
    tp_acc = 0
    #テストデータが1001~13000 トレインデータが13001~23000
    train_point = 1000
    change_point = 13000
    test_point = 23000
    #train_size = change_point-train_point#12000
    test_size = test_point-change_point#10000
    S = []
    T = []
    #学習
    for i in range(train_point,change_point):#12000
      #順伝播
      u = input[:,i].reshape(16,1).float()
      sp_pre, tp_pre, state = model.forward(u, weight_in, weight_res_in,weight_res_out, b_in, b_res)
      if i == train_point:
        X = state
      else:
        X = torch.cat((X,state),1)

    #正解ラベルの作成
    sp_train_ans = sp_ans[train_point:change_point,:]
    tp_train_ans = tp_ans[train_point:change_point,:]
    #リッジ回帰
    model.sp_weight_out = model.ridge(X.T, model.sp_weight_out, sp_train_ans.float()).float()
    model.tp_weight_out = model.ridge(X.T, model.tp_weight_out, tp_train_ans.float()).float()
    #学習確認用のプリント
    #print(f'sp_weight_out{sp_weight_out}')
    #print(f'tp_weight_out{tp_weight_out}')  
    
    #テスト
    for i in range(change_point,test_point):#10000
      u = input[:,i].reshape(16,1).float()
      sp_pre, tp_pre, state = model.forward(u, weight_in, weight_res_in,weight_res_out, b_in, b_res)
      #sp_pre = model.softmax_func(sp_pre)
      #tp_pre = model.softmax_func(tp_pre)
      #sp_pre = model.sigmoid_func(sp_pre)
      #tp_pre = model.sigmoid_func(tp_pre)
      if i == train_point:
        Y = state
      else:
        Y = torch.cat((X,state),1)
      if i == change_point:
        S = sp_pre
        T = tp_pre
      else:
        S = torch.cat((S,sp_pre),1)
        T = torch.cat((T,tp_pre),1)
      #空間情報の正解率の算出
      if torch.argmax(sp_pre) == torch.argmax(sp_ans[i,:]):
        sp_acc += 1
      #時間情報の正解率の算出
      if torch.argmax(tp_pre) == torch.argmax(tp_ans[i,:]):
        tp_acc += 1
    sp_acc = sp_acc/test_size
    tp_acc = tp_acc/test_size
    #正解ラベルの作成
    sp_test_ans = sp_ans[change_point:test_point,:]
    tp_test_ans = tp_ans[change_point:test_point,:]
    #クロスエントロピー誤差の計算
    S = S.T
    T = T.T
    loss_func = nn.CrossEntropyLoss()
    #loss_func = nn.MSELoss()
    #loss_func = model.cross_entropy_error
    #print(len(sp_test_ans.float()))
    #print(sp_test_ans.float().shape)
    sp_max_ans = torch.argmax(sp_test_ans, 1)
    tp_max_ans = torch.argmax(tp_test_ans, 1)
    sp_loss = loss_func(S, sp_max_ans)
    tp_loss = loss_func(T, tp_max_ans)
    #相互情報量の算出
    h_in_x, h_in_y, h_out_x, h_out_y = self.mutial_info(Y, sp_max_ans, tp_max_ans)
    #結果の表示
    sp_loss_str = f'sp_loss{sp_loss:.4f}'
    tp_loss_str = f'tp_loss{tp_loss:.4f}'
    sp_accuracy_str = f'sp_accuracy{sp_acc:.4f}'
    tp_accuracy_str = f'tp_accuracy{tp_acc:.4f}'
    print(f'----{sp_accuracy_str} | {sp_loss_str} | {tp_accuracy_str} | {tp_loss_str}----')
    acc_data = sp_acc, tp_acc
    loss_data = sp_loss, tp_loss
    mutial_info_data = h_in_x, h_in_y, h_out_x, h_out_y
    weight_data = weight_res_in,weight_res_out, b_res, model.sp_weight_out, model.tp_weight_out
    return weight_data, acc_data, loss_data, mutial_info_data
    
  def mutial_info(self, Y, sp_max_ans, tp_max_ans):
    mutial_range = -150
    in_neurons = Y[:16,:]
    out_neurons = Y[-16:,:]
    h_in_x = []
    h_in_y = []
    h_out_x = []
    h_out_y = []
    bins=8
    range_x1=(-1,1)
    for j in range(16):
      n_in = in_neurons[j,mutial_range:]
      n_out = out_neurons[j,mutial_range:]
      _,bins_x1  = np.histogram(n_in, bins,range_x1)
      _,bins_x1  = np.histogram(n_out, bins,range_x1)
      n_in_mutial = np.digitize(n_in, bins_x1)
      n_out_mutial = np.digitize(n_out, bins_x1)
      n_in_x_mutial = sklearn.metrics.mutual_info_score(n_in_mutial,sp_max_ans[mutial_range:])
      n_in_y_mutial = sklearn.metrics.mutual_info_score(n_in_mutial,tp_max_ans[mutial_range:])
      n_out_x_mutial = sklearn.metrics.mutual_info_score(n_out_mutial,sp_max_ans[mutial_range:])
      n_out_y_mutial = sklearn.metrics.mutual_info_score(n_out_mutial,tp_max_ans[mutial_range:])
      h_in_x.append(n_in_x_mutial)
      h_in_y.append(n_in_y_mutial)
      h_out_x.append(n_out_x_mutial)
      h_out_y.append(n_out_y_mutial)
    return h_in_x,h_in_y,h_out_x,h_out_y

if __name__ == '__main__':
 #初期化
 inputdata = input_np.make_data
 model = ESN_Model.Binde_ESN_Model()
 training = Ridge_train(model, inputdata)
 weight_in, b_in = model.setup_in()
 weight_res, b_res = model.setup_res()
 name = "test_ana"
 #学習
 weight_data, acc_data, loss_data, mutial_info_data = training.main(weight_in, weight_res, b_in, b_res) 
 sp_acc, tp_acc = acc_data
 sp_loss, tp_loss = loss_data
 loss = (sp_loss + tp_loss)/2
 acc = (sp_acc + tp_acc)/2
 print(acc)
 print(loss)
 weight_res, b_res, sp_weight_out, tp_weight_out = weight_data
 h_in_x, h_in_y, h_out_x, h_out_y = mutial_info_data
 #保存
 ana = Analysis.Analysis(name)
 ana.save_to_weight(weight_res, b_res,sp_weight_out,tp_weight_out)
 ana.save_to_mutial(h_in_x, h_in_y, h_out_x, h_out_y)
 #ana.save_to_data(acc_data, loss_data)

