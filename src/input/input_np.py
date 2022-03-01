import torch
import matplotlib.pyplot as plt
import math
import random
import os

#データ作成用
def make_pattern(t_long,h_long,patterns):
  sp_patt = []
  tp_patt = []
  ts = torch.arange(0.0, t_long , 1.0)
  cols = [torch.Tensor() for _ in range(h_long)]
  for p in patterns:
    tp, sp = p
    a = -1
    for i in range(h_long):
      if i % sp == 0:
        a *= -1
      data = a * torch.cos(2.0*math.pi * tp *ts)
      cols[i] = torch.cat([cols[i], data])
    for j in range(list(ts.shape)[0]):
      #正解データの作成
      if sp == 2:
        spot = 0
        sp_patt.append(spot)
      elif sp == 4:
        spot = 1
        sp_patt.append(spot)
      elif sp == 8:
        spot = 2
        sp_patt.append(spot)
      #正解データの作成
      if tp == 1/4:
        spot = 0
        tp_patt.append(spot)
      elif tp ==1/8:
        spot = 1
        tp_patt.append(spot)
      elif tp ==1/16:
        spot = 2
        tp_patt.append(spot)
  sp_one_hot = torch.nn.functional.one_hot(torch.tensor(sp_patt), num_classes=3)
  tp_one_hot = torch.nn.functional.one_hot(torch.tensor(tp_patt), num_classes=3)
  patt = torch.cat((sp_one_hot,tp_one_hot),1)
  cols = torch.stack(cols).view(h_long, -1)
  return cols, patt, sp_one_hot, tp_one_hot

def make_seed_patt():
  patterns = []
  spatial_patterns = [2, 4, 8]
  temporal_patterns = [1/4, 1/8, 1/16]
  for sp in spatial_patterns:
    for tp in temporal_patterns:
      patterns.append((tp, sp))
  random.shuffle(patterns)
  return patterns

def make_train(t_long=15,h_long=16,batch_size=10):
  traindata = [torch.Tensor() for _ in range(batch_size)]
  ans = []
  for i in range(batch_size):
    patterns = make_seed_patt()
    #学習データの作成
    change_switch = random.randint(0,1)
    if change_switch == 0:
      trainpatt, train, _, _ = make_pattern(t_long*2,h_long,patterns[0:1])
      traindata[i] = torch.cat([traindata[i], trainpatt])
      ans.append(train)
    else:
      trainpatt, train, _, _ = make_pattern(t_long,h_long,patterns[0:2])
      traindata[i] = torch.cat([traindata[i], trainpatt])
      ans.append(train)
  traindata = torch.stack(traindata).view(batch_size, h_long, -1)
  train_ans = torch.stack(ans).view(batch_size,-1, 6)
  #学習データをgpuに
  traindata = traindata.cuda()
  train_ans = train_ans.cuda()
  return traindata, train_ans

def make_data(t_long=30, h_long=16):
  patterns = []
  for i in range(86):
    pattern = make_seed_patt()
    patterns[len(patterns):len(patterns)] = pattern
  #シャッフルの後，評価データの作成
  data,_ , sp_test, tp_test = make_pattern(t_long, h_long, patterns)
  return data, sp_test, tp_test

if __name__ == '__main__':
  data, sp_test, tp_test = make_data()
  #print(train.shape)
  #print(train_ans.shape)
  #print(testdata.shape)
  #print(sp_test.shape)
  data_name = str(os.path.dirname(os.path.abspath(__file__)))+'/img/input'
  plt.figure()
  plt.imshow(data[:,1000:1200])
  plt.savefig(f''+data_name+'.pdf')
  plt.savefig(f''+data_name+'.svg')