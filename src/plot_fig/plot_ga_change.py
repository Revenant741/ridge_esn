import numpy as np
import random
import copy
from multiprocessing import Process, Queue
from math import exp,cos,sin,log
import matplotlib.pyplot as plt
import model
from inputdata import InputData

#リザバー層の計算
#ESN_IN = 𝑊𝑖𝑛*U𝑡−1 + X𝑡−1*𝑊
#ESN_IN = 32*16@16*1 + 32*32@32*1 
#state = f｛(1-δ)X𝑡−1 + δ (ESN_IN)+noise｝
def reservoir(input, weight_in, state, weight_res, leaky):
  bias = 0.1*(2*random.uniform(1, 32)-1)
  noise = 2*random.uniform(32, 1)-1
  ESN_IN = weight_in @ input   + weight_res @ state 
  state = np.tanh((1 - leaky) * state + leaky *(ESN_IN) +bias )+noise
  return state

#w=(X^Tx+C*E)^-1:X^T*y
def ridge(state,weight_out,ans):
  C = 10
  E = np.eye(32)
  weight_out = ((state.T@state+C*E)**-1)@(state.T@ans)
  #リッジ回帰の拘束
  bind = np.concatenate([np.zeros((3,16)), np.ones((3,16))],1)
  weight_out *= bind.T
  return weight_out.T

def main(weight_in, weight_res, sp_weight_out, tp_weight_out, state):
  #入力データを取得
  input, sp_ans, tp_ans = InputData.makeinput() 
  
  #パラメータの設定
  size_in = 16
  size_res = 32
  size_out = 6
  leaky = 0.8
  sp_acc = 0
  tp_acc = 0
  #テストデータが1001~13000 トレインデータが13001~23000
  train_size = 13000
  test_size = 23000

  '''
  print('{}=weight_in'.format(weight_in))
  print('{}=weight_res'.format(weight_res))
  print('{}=weight_out'.format(sp_weight_out))
  '''
  X = []
  #学習
  for i in range(1001,train_size):#11999
    #順伝播
    state = reservoir(input[:,i].reshape(16,1), weight_in, state, weight_res, leaky)
    X.append(state.reshape([32]))
    sp_pre = sp_weight_out @ state
    tp_pre = tp_weight_out @ state
  X = np.array(X)
  #リッジ回帰
  sp_weight_out = ridge(X, sp_weight_out, sp_ans[1001:13000].T.reshape([11999, 3]))
  tp_weight_out = ridge(X, tp_weight_out, tp_ans[1001:13000].T.reshape([11999, 3]))
  #重みの修正確認
  #print('{}=weight_out'.format(sp_weight_out))
  #テスト
  for i in range(train_size+1,test_size):
    state = reservoir(input[:,i].reshape(16,1), weight_in, state, weight_res, leaky)
    sp_pre = sp_weight_out @ state
    tp_pre = tp_weight_out @ state
    #正解率の算出
    if np.argmax(sp_pre) == np.argmax(sp_ans[i,:].reshape(3,1)):
      sp_acc += 1
    if np.argmax(tp_pre) == np.argmax(tp_ans[i,:].reshape(3,1)):
      tp_acc += 1
  sp_acc = sp_acc/len(range(1001,train_size))
  tp_acc = tp_acc/len(range(1001,train_size))
  accuracy = (sp_acc+tp_acc)/2
  return sp_acc, tp_acc

def first_gene(ind,pop,weight_in, sp_weight_out, tp_weight_out, state):
  #第一世代の作成
  for i in range(pop):
    weight_res = model.setup_res()
    sp_acc, tp_acc = main(weight_in, weight_res, sp_weight_out, tp_weight_out, state)
    ind.append((sp_acc,tp_acc))
  return ind


if __name__ == '__main__':
  pop = 100 #初期個体
  ind = []
  #リザバー以外の重みの固定
  weight_in, sp_weight_out, tp_weight_out, state = model.setup()
  #第一世代の実行
  ind = first_gene(ind,pop,weight_in, sp_weight_out, tp_weight_out, state)

  T = []
  S = []
  for sp_acc, tp_acc in ind:
    print(f'accuracy{tp_acc}')
    print(f'accuracy{sp_acc}')
    T.append(tp_acc)
    S.append(sp_acc)

  plt.xlabel("Accuracy")
  plt.ylabel("Number of individuals")
  plt.hist(T,label="time")
  plt.hist(S,label="space")
  plt.legend(loc='upper right')
  plt.savefig('model_right.png')

