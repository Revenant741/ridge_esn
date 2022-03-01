import argparse
import numpy as np
import matplotlib.pyplot as plt
import random
import copy
import csv
import pickle
from multiprocessing import Process, Queue
from math import exp,cos,sin,log
from input import input_np
import train as train
from model import binde_esn_model as ESN_Model
from my_def import Analysis

def add_arguments(parser):
  parser.add_argument('--name', type=str, default='ga_ridge_test', help='save_file_name')
  parser.add_argument('--pop', type=int, default=20, help='pop_model_number')
  parser.add_argument('--survivor', type=int, default=10, help='pop_model_number')
  parser.add_argument('--mutate_rate', default=0.25, help='mutate_rate')
  parser.add_argument('--generation', type=int, default=1000, help='generation')
  parser.add_argument('--gene_length', default=16, help='pop_model_number')
  #parser.add_argument('--Leaky', type=bool, default=True, help='Use_serching_parameter?')
  #python3 src/esn_ga_train.py  --pop 10 --survivor 4 --name 'ga_test'
  #python3 src/esn_ga_train.py --name 'ga_ridge_e20_p20_l10'
  #python3 src/esn_ga_train.py --name 'ga_ridge_2'
  #python3 src/esn_ga_train.py --name 'ga_ridge_3'
  #python3 src/esn_ga_train.py --name 'ga_ridge_4'
  #REvenant741
  #--name 'ga_ridge_model_1'
  
def first_gene(model,inputdata,ind, pop, weight_in,b_in,b_res):
  #第一世代の作成
  for i in range(pop):
    #個体生成
    model = ESN_Model.Binde_ESN_Model()
    #print(f'個体{i+1}')
    weight_res1, weight_res2,weight_res3,weight_res4,_ = model.setup_res_all()
    #モデルの実行準備
    ridge_learn = train.Ridge_train(model,inputdata)
    #リッジ回帰で学習，重み,誤差，精度を返して来る．
    weight_data, acc_data, loss_data, mutual_info_data = ridge_learn.main(weight_in,weight_res1,weight_res2,weight_res3,weight_res4, b_in, b_res) 
    sp_acc, tp_acc = acc_data
    sp_loss, tp_loss = loss_data
    h_in_x, h_in_y, h_out_x, h_out_y = mutual_info_data
    loss = (sp_loss + tp_loss)/2
    acc = (sp_acc + tp_acc)/2
    ind.append((acc, loss, weight_data, sp_acc, tp_acc, sp_loss, tp_loss,h_in_x, h_in_y, h_out_x, h_out_y))
  return ind

def next_gene(model,inputdata,ind, pop, weight_in,b_in,elites1,elites2,elites3,elites4,b_res):
  #第g世代の作成
  for i in range(pop):
    #個体生成
    model = ESN_Model.Binde_ESN_Model()
    weight_res1 = elites1[i]
    weight_res2 = elites2[i]
    weight_res3 = elites3[i]
    weight_res4 = elites4[i]
    #モデルの実行，重み,誤差，精度を返して来る．
    ridge_learn = train.Ridge_train(model,inputdata)
    weight_data, acc_data, loss_data, mutual_info_data = ridge_learn.main(weight_in,weight_res1,weight_res2,weight_res3,weight_res4,b_in,b_res) 
    sp_acc, tp_acc = acc_data
    sp_loss, tp_loss = loss_data
    h_in_x, h_in_y, h_out_x, h_out_y = mutual_info_data
    loss = (sp_loss + tp_loss)/2
    acc = (sp_acc + tp_acc)/2
    ind.append((acc, loss, weight_data, sp_acc, tp_acc, sp_loss, tp_loss,h_in_x, h_in_y, h_out_x, h_out_y))
  return ind

#選択，生き残る個体を決める関数
def evalution(ind):
  #0で精度、1で誤差，Falseで小さい順，Trueで大きい順
  ind = sorted(ind, key=lambda x:x[1], reverse=False)
  #ind = sorted(ind, key=lambda x:x[0], reverse=True)
  return ind
#二点交叉
def tow_point_crossover(parent1, parent2,gene_length):
  child1 = copy.deepcopy(parent1)
  r0 = random.randint(0,15)
  r1 = random.randint(0,gene_length-1)
  r2 = random.randint(r1,gene_length)
  child1[r0,r1:r2]= parent2[r0,r1:r2]
  return child1
#突然変異
def mutate(parent,gene_length):
  child = copy.deepcopy(parent)
  for i in range(10):
    r1 = random.randint(0, gene_length-1)
    r2 = random.randint(0, gene_length-1)
    change = random.random()
    if child[r1][r2] == 0:
      child[r1][r2] = change
    else:
      child[r1][r2] = 0
  return child

def ga_train(args):
  ind = []
  #パラメータ
  gene_length = args.gene_length#切り替わる遺伝子の長さ
  elites1 = []
  elites2 = []
  elites3 = []
  elites4 = []
  inputdata = input_np.make_data
  name = args.name
  #記録用の変数
  SP_A = []
  TP_A = []
  SP_L = []
  TP_L = []
  W_res1 = []
  W_res2 = []
  W_res3 = []
  W_res4 = []
  SP_W = []
  TP_W = []
  G = []
  W = []
  X_IN ,Y_IN ,X_OUT, Y_OUT = [],[],[],[]
  #リザバー以外の重みの固定
  model = ESN_Model.Binde_ESN_Model()
  weight_in, b_in = model.setup_in()
  _,_,_,_, b_res = model.setup_res_all()
  ana = Analysis.Analysis(name)
  #b_resも保存対象に連絡
  ana.save_in_layer_para(weight_in, b_in, b_res)
  #print(weight_in)
  #第一世代の精度と重み
  print(f'世代1')
  first_pop = args.pop
  ind = first_gene(model,inputdata,ind, first_pop, weight_in,b_in,b_res)
  #第g世代の作成
  for g in range(args.generation):
    print(f'世代{g+2}')
    #エリートを選択
    ind = evalution(ind)
    #選択,indの初期化
    survival = ind[0:args.survivor]
    ind = survival
    rank = 0
    #記録用の処理
    for acc, loss, weight_data,sp_acc, tp_acc, sp_loss, tp_loss,h_in_x, h_in_y, h_out_x, h_out_y in survival:
      #リザバー層重みを取得
      weight_res1,weight_res2,weight_res3,weight_res4, b_res, sp_weight_out, tp_weight_out = weight_data
      elites1.append(weight_res1)
      elites2.append(weight_res2)
      elites3.append(weight_res3)
      elites4.append(weight_res4)
      rank += 1
      print(f'--第{rank}位--空間精度={sp_acc*100:.1f}%-時間精度={tp_acc*100:.1f}%---誤差={loss:.4f}-')
      SP_A.append(sp_acc)
      TP_A.append(tp_acc)
      SP_L.append(sp_loss)
      TP_L.append(tp_loss)
      G.append(g+1)
      W_res1.append(weight_res1)
      W_res2.append(weight_res2)
      W_res3.append(weight_res3)
      W_res4.append(weight_res4)
      #B_res.append(b_res)
      SP_W.append(sp_weight_out)
      TP_W.append(tp_weight_out)
      X_IN.append(h_in_x)
      Y_IN.append(h_in_y)
      X_OUT.append(h_out_x)
      Y_OUT.append(h_out_y)
    #記録
    #if G[-1]%10 == 0:
    ana.save_to_ga_binde_data(SP_A,TP_A,SP_L,TP_L,G,W_res1,W_res2,W_res3,W_res4,SP_W,TP_W)
    ana.save_to_ga_mutual_data(X_IN ,Y_IN ,X_OUT, Y_OUT)
    #次世代に向けた処理
    #変化無しの個体
    #交叉
    while len(elites1) < first_pop:
      m1 = random.randint(0,len(elites1)-1)#親となる個体の決定
      m2 = random.randint(0,len(elites1)-1)#親となる個体の決定
      child1 = tow_point_crossover(elites1[m1],elites1[m2],gene_length)#交叉処理
      child2 = tow_point_crossover(elites2[m1],elites2[m2],gene_length)#交叉処理
      child3 = tow_point_crossover(elites3[m1],elites3[m2],gene_length)#交叉処理
      child4 = tow_point_crossover(elites4[m1],elites4[m2],gene_length)#交叉処理
      elites1.append(child1)
      elites2.append(child2)
      elites3.append(child3)
      elites4.append(child4)
      #突然変異
      if random.random() < args.mutate_rate:
        m = random.randint(0,len(elites1)-1)#突然変異する個体を選択
        child1 = mutate(elites1[m],gene_length)
        child2 = mutate(elites2[m],gene_length)
        child3 = mutate(elites3[m],gene_length)
        child4 = mutate(elites4[m],gene_length)
      elites1.append(child1)
      elites2.append(child2)
      elites3.append(child3)
      elites4.append(child4)
    #次世代の評価
    #args.pop = first_pop-args.survivor
    ind = next_gene(model,inputdata,ind, args.pop, weight_in,b_in, elites1,elites2,elites3,elites4,b_res)
    #初期化
    elites1 = []
    elites2 = []
    elites3 = []
    elites4 = []
  
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  add_arguments(parser)
  args = parser.parse_args()
  print(args)
  #パラメータ
  ga_train(args)