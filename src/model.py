import numpy as np
from numpy.random import *

class BindeESN():
  def __init__(self, size_in, size_res, size_out,leaky):
    self.size_in = size_in
    self.size_res = size_res
    self.size_out = size_out
    self.weight_in = np.ones(size_in, size_res)
    self.weight_res = np.ones(size_res, size_res)
    self.weight_out = np.ones(size_res, size_out)
    self.bias = np.ones(size_res)
    self.state = np.ones(size_res)
    self.noise = np.ones(size_res)
    self.reset_parameters()
    self.Linear = Bindlinear(size_res, size_out)
    self.leaky = leaky

  def reset_parameters(self):
    self._reset_weight_in()
    self._reset_weight_res()
    self._reset_bias()
    self._reset_state()
    self._reset_noise()

  def _reset_weight_in(self):
    unconnect = np.zeros((16,16))
    connect = np.ones((16,16))
    bind = np.concatenate([connect,unconnect],1)
    self.weight_in = self.weight_in*bind

  def _reset_weight_res(self):
    adjency = rand(size_res,size_res) 
    self.weight_res *= adjency

  def _reset_bias(self):
    adjency = rand(size_res,size_res)
    self.bias *= adjency
    #baiasの初期化処理


  def _reset_state(self):
    self.state = np.zeros(size_res)

  def _reset_noise(self):
    self.noise = random.uniform(1e-7,1e-3)

  def _reset_weight_out(self):
    unconnect = np.zeros((16,3))
    connect = np.ones((16,3))
    bind = np.concatenate([unconnect,connect],0)
    self.weight_in = self.weight_in*bind

  def res_state(self, x):

    tilde_x = x @ self.weight_in + self.state @self.weight_res +self.bias

    self.state = np.tanh((1 - self.leaky) * self.state + self.leaky *tilde_x + self.noise)

    return forward(self.state), forward(self.state)

  def forward(self, input):
    out_put = input*self.weight_out+self.bias
    return out_put





     