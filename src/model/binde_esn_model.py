import math
#from os import PRIO_PGRP
import numpy as np
import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt
from input import input_np
from input import inputdata

class Binde_ESN_Model:
  def __init__(self):
    super().__init__()
    self.size_in = 16
    self.size_middle_in = 16
    self.size_middle_out = 16
    self.size_out = 3
    self.setup()

  #リザバー層の内部状態と出力層の初期化
  def setup(self):
    self.state_in = self.reset_state()
    self.state_out = self.reset_state()
    self.sp_weight_out = self.reset_weight_out()
    self.tp_weight_out = self.reset_weight_out()

  #リザバー層の内部状態を初期化
  def reset_state(self):#16*1
    #state = torch.zeros((self.size_middle_in,1))
    #state = np.random.random_sample((self.size_middle_in,1))
    #state = torch.from_numpy(state.astype(np.float32)).clone()
    state = torch.rand((self.size_middle_in,1))
    return state

  #出力層の重みの初期化3*16
  def reset_weight_out(self):
    #weight_out = np.random.random_sample((self.size_out,self.size_middle_in))
    #weight_out = torch.from_numpy(weight_out.astype(np.float32)).clone()
    weight_out = torch.rand((self.size_out,self.size_middle_in))
    return weight_out

  #入力層の初期化16*16
  def setup_in(self):
    #weight_in = np.random.random_sample((self.size_middle_in, self.size_in))
    #weight_in = torch.from_numpy(weight_in.astype(np.float32)).clone()
    weight_in = torch.rand((self.size_middle_in, self.size_in))
    #biasの初期値
    b_in = torch.Tensor(self.size_middle_in,1)
    b_in = self.reset_bias(weight_in,b_in)
    return weight_in, b_in

  #リザバー層重み全ての初期化
  #16*16が4つで実質32＊32
  def setup_res_all(self):
    weight_res1 = self.setup_res()
    weight_res2 = self.setup_res()
    weight_res3 = self.setup_res()
    weight_res4 = self.setup_res()
    #biasの初期値
    b_res = torch.Tensor(self.size_middle_in,1)
    b_res = self.reset_bias(weight_res1,b_res)
    return weight_res1,weight_res2,weight_res3,weight_res4,b_res

  def setup_res(self):
    #weight_res_out = np.random.random_sample((self.size_middle_out, self.size_middle_out))
    #weight_res_out = torch.from_numpy(weight_res_out.astype(np.float32)).clone()
    weight_res_out = torch.rand((self.size_middle_out, self.size_middle_out))
    for i in range(4):
      adjency = torch.tensor([random.randint(0, 1) for _ in range(self.size_middle_out**2)])
      weight_res_out *= adjency.reshape(self.size_middle_out, self.size_middle_out)
    #print(weight_res_out)
    return weight_res_out

  def reset_bias(self, weight,bias):
    #バイアスの初期化
    #fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
    bound = 1 / math.sqrt(5)
    nn.init.uniform_(bias, -bound, bound)
    bias =torch.rand((self.size_middle_out,1))
    return bias

  #順伝搬の計算
  def forward(self, u, weight_in, weight_res1,weight_res2,weight_res3,weight_res4, b_in, b_res):
    self.state_in = torch.tanh(torch.matmul(weight_in,u) + b_in + torch.matmul(weight_res1,self.state_in)+torch.matmul(weight_res2,self.state_out))
    self.state_out = torch.matmul(weight_res3,self.state_in) + torch.matmul(weight_res4,self.state_out)
    self.state_in = torch.tanh(self.state_in)
    self.state_out = torch.tanh(self.state_out)
    sp_pre = torch.matmul(self.sp_weight_out,self.state_out+ b_res)
    tp_pre = torch.matmul(self.tp_weight_out,self.state_out+ b_res)
    return sp_pre, tp_pre, self.state_out

  #リッジ回帰の計算
  #w=(X^Tx+C*E)^-1:X^T*y
  def ridge(self, state,weight_out,ans):
    C = 1
    E = torch.eye(16)
    weight_out = torch.matmul(torch.inverse(torch.matmul(state.T,state)+C*E),(torch.matmul(state.T,ans)))
    return weight_out.T

  #クロスエントロピー誤差
  def cross_entropy_error(self, out_put, ans):
    delta = 1e-7
    data_size = len(out_put)
    loss = -torch.sum(ans*torch.log(out_put+delta))/data_size
    return loss

  def softmax_func(self,x):
      exp_x = torch.exp(x)
      return exp_x/torch.sum(exp_x)

  def sigmoid_func(self,x):
    return 1/(1+torch.exp(-x))
