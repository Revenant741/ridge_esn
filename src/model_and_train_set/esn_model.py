import math
import numpy as np
import random
import matplotlib.pyplot as plt
from input import input_np
from input import inputdata

#入力層の重みの初期化16*16
def reset_weight_in():
  weight_in = np.random.random_sample((16, 16))
  return weight_in

#出力層の重みの初期化3*16
def reset_weight_out():
  weight_out = np.random.random_sample((16, 16))
  return weight_out

#reservoir重みの初期化 16*16
def reset_weight_res():
    weight_res = np.random.random_sample((16, 16))
    adjency = np.array([random.randint(0, 1) for _ in range(16**2)])
    weight_res *= adjency.reshape(16, 16)
    return weight_res

#リザバー層の内部状態を初期化
def reset_state_in():#16*1
    state_in = np.zeros((16,1))
    return state_in

#リザバー層の内部状態を初期化
def reset_state_out():#16*1
    state_out = np.zeros((16,1))
    return state_out

#リザバー層の計算
#ESN_IN = 𝑊𝑖𝑛*U𝑡−1 + X𝑡−1*𝑊
#ESN_IN = 32*16@16*1 + 32*32@32*1 
#state = f｛(1-δ)X𝑡−1 + δ (ESN_IN)｝
def reservoir(input, weight_in, state_in, state_out, weight_res_in, weight_res, weight_res_out):
  state_in = weight_in @ input   + weight_res_in @ state_in
  state_out = np.tanh(weight_res @ state_in   + weight_res_out @ state_out)
  return state_out


#リッジ回帰の計算
#w=(X^Tx+C*E)^-1:X^T*y
def ridge(state,weight_out,ans):
  C = 1
  E = np.eye(16)
  weight_out = ((state.T@state+C*E)**-1)@(state.T@ans)
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
  state_in = reset_state_in()
  state_out = reset_state_out()
  return weight_in, sp_weight_out, tp_weight_out, state_in, state_out

#リザバー層の初期化
def setup_res():
  weight_res_in = reset_weight_res()
  weight_res = reset_weight_res()
  weight_res_out = reset_weight_res()
  return weight_res_in, weight_res, weight_res_out


def softmax_func(x):
    exp_x = np.exp(x)
    return exp_x/np.sum(exp_x)

#main関数
def main(weight_in, weight_res_in, weight_res, weight_res_out, state_in, state_out, sp_weight_out, tp_weight_out):
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
  for i in range(train_point,change_point):#12000
    #順伝播
    state = reservoir(input[:,i].reshape(16,1), weight_in, state_in, state_out, weight_res_in, weight_res, weight_res_out)
    sp_pre = np.tanh(sp_weight_out @ state)
    tp_pre = np.tanh(tp_weight_out @ state)
    X.append(state.reshape([16]))
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
  
  #テスト
  for i in range(change_point,test_point):#10000
    state = reservoir(input[:,i].reshape(16,1), weight_in, state_in, state_out, weight_res_in, weight_res, weight_res_out)
    sp_pre = sp_weight_out @ state
    tp_pre = tp_weight_out @ state
    S.append(sp_pre)
    T.append(tp_pre)
    #空間情報の正解率の算出
    if np.argmax(sp_pre) == np.argmax(sp_ans[i,:]):
      sp_acc += 1
    #時間情報の正解率の算出
    if np.argmax(tp_pre) == np.argmax(tp_ans[i,:]):
      tp_acc += 1
  sp_acc = sp_acc/test_size
  tp_acc = tp_acc/test_size
  #正解ラベルの作成
  sp_test_ans = sp_ans[change_point:test_point,:]
  tp_test_ans = tp_ans[change_point:test_point,:]
  #クロスエントロピー誤差の計算
  S = np.array(S)
  T = np.array(T)
  S = S.reshape([test_size, 3])
  T = T.reshape([test_size, 3])
  sp_loss = cross_entropy_error(softmax_func(S), sp_test_ans, test_size)
  tp_loss = cross_entropy_error(softmax_func(T), tp_test_ans, test_size)
  #sp_loss = mean_squared_error(S, sp_test_ans, test_size)
  #tp_loss = mean_squared_error(T, tp_test_ans, test_size)
  #結果の表示
  sp_loss_str = f'sp_loss{sp_loss}'
  tp_loss_str = f'tp_loss{tp_loss}'
  sp_accuracy_str = f'sp_accuracy{sp_acc:.4f}'
  tp_accuracy_str = f'tp_accuracy{tp_acc:.4f}'
  print(f'----{sp_accuracy_str} | {sp_loss_str} | {tp_accuracy_str} | {tp_loss_str}----')
  return weight_res_in, weight_res, weight_res_out, sp_acc, tp_acc, sp_loss, tp_loss

if __name__ == '__main__':
 #初期化
 weight_in, sp_weight_out, tp_weight_out, state_in, state_out = setup()
 weight_res_in, weight_res, weight_res_out = setup_res()
 #学習
 weight_res_in, weight_res, weight_res_out, sp_acc, tp_acc, sp_loss, tp_loss = main(weight_in, weight_res_in, weight_res, weight_res_out, state_in, state_out, sp_weight_out, tp_weight_out)

