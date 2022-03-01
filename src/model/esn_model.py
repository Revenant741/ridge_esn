import math
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
    self.size_middle = 32
    self.size_out = 3
    self.setup()

  #入力層の初期化
  def setup_in(self):
    weight_in = np.random.random_sample((self.size_middle, self.size_in))
    weight_in = torch.from_numpy(weight_in.astype(np.float32)).clone()
    bind = torch.cat([torch.ones((self.size_in,self.size_in)), torch.zeros((self.size_in,self.size_in))])
    weight_in *= bind
    #biasの初期値
    b_in = torch.Tensor(self.size_middle,1)
    b_in = self.reset_bias(weight_in,b_in)
    return weight_in, b_in

  #リザバー層の初期化
  def setup_res(self):
    weight_res = np.random.random_sample((self.size_middle, self.size_middle))
    weight_res = torch.from_numpy(weight_res.astype(np.float32)).clone()
    adjency = torch.tensor([random.randint(0, 1) for _ in range(self.size_middle**2)])
    weight_res *= adjency.reshape(self.size_middle, self.size_middle)
    #biasの初期値
    b_res = torch.Tensor(self.size_middle,1)
    b_res = self.reset_bias(weight_res,b_res)
    return weight_res, b_res

  #出力層と内部状態の初期化
  def setup(self):
    self.sp_weight_out = self.reset_weight_out()
    self.tp_weight_out = self.reset_weight_out()
    self.state = self.reset_state()

  #リザバー層の内部状態を初期化
  def reset_state(self):#32*1
      state = torch.zeros((self.size_middle,1))
      return state

  #出力層の重みの初期化3*32
  def reset_weight_out(self):
    weight_out = np.random.random_sample((self.size_out,self.size_middle))
    weight_out = torch.from_numpy(weight_out.astype(np.float32)).clone()
    bind = torch.cat([torch.zeros((self.size_out,self.size_in)), torch.ones((self.size_out,self.size_in))],1)
    weight_out *= bind
    return weight_out

  def reset_bias(self, weight,bias):
    #バイアスの初期化
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
    bound = 1 / math.sqrt(fan_in)
    nn.init.uniform_(bias, -bound, bound)
    return bias

  #順伝搬の計算
  #ESN_IN = 𝑊𝑖𝑛*U𝑡−1 + X𝑡−1*𝑊
  #ESN_IN = 32*16@16*1 + 32*32@32*1
  #state = f｛(1-δ)X𝑡−1 + δ (ESN_IN)｝
  def forward(self, u, weight_in, weight_res, b_in, b_res):
    #print(torch.matmul(weight_in,u) + b_in)
    self.state = torch.tanh(torch.matmul(weight_in,u) + b_in + torch.matmul(weight_res,self.state)) + b_res
    sp_pre = torch.matmul(self.sp_weight_out,self.state)
    tp_pre = torch.matmul(self.tp_weight_out,self.state)
    return sp_pre, tp_pre, self.state

  #リッジ回帰の計算
  #w=(X^Tx+C*E)^-1:X^T*y
  def ridge(self, state,weight_out,ans):
    C = 1
    E = torch.eye(32)
    weight_out = torch.matmul(torch.inverse(torch.matmul(state.T,state)+C*E),(torch.matmul(state.T,ans)))
    #リッジ回帰の拘束
    #bind = torch.cat([torch.zeros((3,16)), torch.ones((3,16))],1)
    #weight_out *= bind.T
    #print(weight_out)
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
