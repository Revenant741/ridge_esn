import math
import numpy as np
import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt
from input import input_np
from input import inputdata
import sklearn.metrics

#入力層の重みの初期化32*16
def reset_weight_in():
    weight_in = torch.Tensor(32,16)
    reset_parameters(weight_in)
    weight_in = weight_in.to('cpu').detach().numpy().copy()
    bind = np.concatenate([np.ones((16,16)), np.zeros((16,16))])
    weight_in *= bind
    return weight_in

#リザバー層重みの初期化32*32
def reset_weight_res(i):
    weight_res = torch.Tensor(32,32)
    reset_parameters(weight_res)
    weight_res = weight_res.to('cpu').detach().numpy().copy()
    adjency = np.array([random.randint(0, 1) for _ in range(32**2)])
    weight_res *= adjency.reshape(32, 32)
    return weight_res

#出力層の重みの初期化3*32
def reset_weight_out():
    weight_out = torch.Tensor(3,32)
    reset_parameters(weight_out)
    weight_out = weight_out.to('cpu').detach().numpy().copy()
    bind = np.concatenate([np.zeros((3,16)), np.ones((3,16))],1)
    weight_out *= bind
    return weight_out

def reset_parameters(weight):
  #重みの初期値
  nn.init.kaiming_uniform_(weight, a=math.sqrt(5))

def reset_bias(weight,bias):
  #バイアスの初期値
  weight = torch.from_numpy(weight).clone()
  fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
  bound = 1 / math.sqrt(fan_in)
  nn.init.uniform_(bias, -bound, bound)
  weight = weight.to('cpu').detach().numpy().copy()

#リザバー層の内部状態を初期化
def reset_state():#32*1
    state = np.zeros((32,1))
    return state

#リザバー層の計算
#ESN_IN = 𝑊𝑖𝑛*U𝑡−1 + X𝑡−1*𝑊
#ESN_IN = 32*16@16*1 + 32*32@32*1
#state = f｛(1-δ)X𝑡−1 + δ (ESN_IN)｝
def reservoir(input, weight_in, state, weight_res, b_in, b_res):
  state = np.tanh(weight_in @ input + b_in + weight_res @ state + b_res)
  return state

#リッジ回帰の計算
#w=(X^Tx+C*E)^-1:X^T*y
def ridge(state,weight_out,ans):
  C = 1
  E = np.eye(32)
  weight_out = np.linalg.inv(state.T@state+C*E)@(state.T@ans)
  #リッジ回帰の拘束
  bind = np.concatenate([np.zeros((3,16)), np.ones((3,16))],1)
  weight_out *= bind.T
  return weight_out.T

#クロスエントロピー誤差
def cross_entropy_error(out_put, ans,data_size):
  delta = 1e-7
  loss = -np.sum(ans*np.log(out_put+delta))/data_size
  return loss

#リザバー層以外の初期化
def setup():
  weight_in = reset_weight_in()
  sp_weight_out = reset_weight_out()
  tp_weight_out = reset_weight_out()
  state = reset_state()
  return weight_in, sp_weight_out, tp_weight_out, state

#リザバー層の初期化
def setup_res(i):
  weight_res = reset_weight_res(i)
  return weight_res

def softmax_func(x):
    exp_x = np.exp(x)
    return exp_x/np.sum(exp_x)

#main関数
def main(weight_in, weight_res, sp_weight_out, tp_weight_out, state):
  #入力データを取得
  #input, sp_ans, tp_ans = input_np.makeinput() 
  input, sp_ans, tp_ans = inputdata.makeinput()
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
  X = []  
  #学習
  #biasの初期値
  b_in = torch.Tensor(32,1)
  b_res = torch.Tensor(32,1)
  reset_bias(weight_in,b_in)
  reset_bias(weight_res,b_res)
  b_in = b_in.to('cpu').detach().numpy().copy()
  b_res = b_res.to('cpu').detach().numpy().copy()
  for i in range(train_point,change_point):#12000
    #順伝播
    state = reservoir(input[:,i].reshape(16,1), weight_in, state, weight_res, b_in, b_res)
    sp_pre = np.tanh(sp_weight_out @ state)
    tp_pre = np.tanh(tp_weight_out @ state)
    X.append(state.reshape([32]))
  X = np.array(X)
  #正解ラベルの作成
  sp_train_ans = sp_ans[train_point:change_point,:]
  tp_train_ans = tp_ans[train_point:change_point,:]
  #リッジ回帰
  sp_weight_out = ridge(X, sp_weight_out, sp_train_ans)
  tp_weight_out = ridge(X, tp_weight_out, tp_train_ans)
  #学習確認用のプリント
  #print(f'sp_weight_out{sp_weight_out}')
  #print(f'tp_weight_out{tp_weight_out}')  
  X = []
  #テスト
  for i in range(change_point,test_point):#10000
    state = reservoir(input[:,i].reshape(16,1), weight_in, state, weight_res, b_in, b_res)
    sp_pre = sp_weight_out @ state
    tp_pre = tp_weight_out @ state
    X.append(state.reshape([32]))
    S.append(sp_pre)
    T.append(tp_pre)
    #空間情報の正解率の算出
    if np.argmax(sp_pre) == np.argmax(sp_ans[i,:]):
      sp_acc += 1
    #時間情報の正解率の算出
    if np.argmax(tp_pre) == np.argmax(tp_ans[i,:]):
      tp_acc += 1
  X = np.array(X)
  sp_acc = sp_acc/test_size
  tp_acc = tp_acc/test_size
  #正解ラベルの作成
  sp_test_ans = sp_ans[change_point:test_point,:]
  tp_test_ans = tp_ans[change_point:test_point,:]
  sp_test_ans = np.argmax(sp_test_ans, 1)
  tp_test_ans = np.argmax(tp_test_ans, 1)
  #結果の表示
  sp_accuracy_str = f'sp_accuracy{sp_acc:.4f}'
  tp_accuracy_str = f'tp_accuracy{tp_acc:.4f}'
  print(f'----{sp_accuracy_str}  | {tp_accuracy_str} ----')

  in_neurons = X[:,:16]
  out_neurons = X[:,-16:]
  h_in_x = []
  h_in_y = []
  h_out_x = []
  h_out_y = []
  bins=8
  range_x1=(-1,1)
  for j in range(16):
    n_in = in_neurons[-150:,j]
    n_out = out_neurons[-150:,j]
    _,bins_x1  = np.histogram(n_in, bins,range_x1)
    _,bins_x1  = np.histogram(n_out, bins,range_x1)
    n_in_mutial = np.digitize(n_in, bins_x1)
    n_out_mutial = np.digitize(n_out, bins_x1)
    n_in_x_mutial = sklearn.metrics.mutual_info_score(n_in_mutial,sp_test_ans[-150:])
    n_in_y_mutial = sklearn.metrics.mutual_info_score(n_in_mutial,tp_test_ans[-150:])
    n_out_x_mutial = sklearn.metrics.mutual_info_score(n_out_mutial,sp_test_ans[-150:])
    n_out_y_mutial = sklearn.metrics.mutual_info_score(n_out_mutial,tp_test_ans[-150:])
    h_in_x.append(n_in_x_mutial)
    h_in_y.append(n_in_y_mutial)
    h_out_x.append(n_out_x_mutial)
    h_out_y.append(n_out_y_mutial)
  return h_in_x,h_in_y,h_out_x,h_out_y

def save_to_mutial(j,h_in_x, h_in_y, h_out_x, h_out_y):
    with open(f'data/h_in_x'+str(j)+'.csv', 'w') as f:
        writer = csv.writer(f)
        for x_1 in h_in_x:
            writer.writerow([x_1])
    with open(f'data/h_in_y'+str(j)+'.csv', 'w') as f:
      writer = csv.writer(f)
      for y_1 in h_in_y:
          writer.writerow([y_1])
    with open(f'data/h_out_x'+str(j)+'.csv', 'w') as f:
        writer = csv.writer(f)
        for x_2 in h_out_x:
            writer.writerow([x_2])
    with open(f'data/h_out_y'+str(j)+'.csv', 'w') as f:
      writer = csv.writer(f)
      for y_2 in h_out_y:
          writer.writerow([y_2])

if __name__ == '__main__':
  for j in range(1):
    in_x = []
    in_y = []
    out_x = []
    out_y = []
    #初期化
    weight_in, sp_weight_out, tp_weight_out, state = setup()
    for i in range(20):
      weight_res = setup_res(i)
      #学習
      h_in_x,h_in_y,h_out_x,h_out_y = main(weight_in, weight_res, sp_weight_out, tp_weight_out, state)
      in_x.append(h_in_x)
      in_y.append(h_in_y)
      out_x.append(h_out_x)
      out_y.append(h_out_y)
    save_to_mutial(j,in_x,in_y,out_x,out_y)
    plt.figure()
    #plt.scatter(in_x,in_y, c='blue',label="input neurons")
    #plt.scatter(out_x,out_y, c='red',label="output neurons")
    plt.scatter(in_x,in_y, c='blue')
    plt.scatter(out_x,out_y, c='blue')
    #plt.xlabel('spatial information',fontsize=15)
    #plt.ylabel('temporal information',fontsize=15)
    #plt.legend(loc='upper right')
    #plt.legend(fontsize=18)
    plt.xlim(0,0.7)
    plt.ylim(0,0.7)
    plt.savefig('img/mutial_info_150'+str(j)+'.png')
    plt.savefig('img/mutial_info_150'+str(j)+'.svg')
    plt.savefig('img/mutial_info_150'+str(j)+'.pdf')

