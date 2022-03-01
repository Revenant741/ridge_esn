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
import train
from model import esn_model as ESN_Model
from model import leaky_esn_model as LeakyESN_Model
from my_def import Analysis

def add_arguments(parser):
  parser.add_argument('--name', type=str, default='ga_ridge_test', help='save_file_name')
  parser.add_argument('--pop', type=int, default=200, help='pop_model_number')
  parser.add_argument('--survivor', type=int, default=20, help='pop_model_number')
  parser.add_argument('--mutate_rate', default=0.25, help='mutate_rate')
  parser.add_argument('--generation', type=int, default=1000, help='generation')
  parser.add_argument('--gene_length', default=31, help='pop_model_number')
  #parser.add_argument('--Leaky', type=bool, default=True, help='Use_serching_parameter?')
  #src/ga.py  --pop 10 --survivor 4 --name 'ga_test'
  #--name 'ga_ridge_1'
  #--name 'ga_ridge_model_1'
  
def first_gene(model,inputdata,ind, pop, weight_in,b_in,b_res):
  #第一世代の作成
  for i in range(pop):
    #print(f'個体{i+1}')
    weight_res, _ = model.setup_res()
    #モデルの実行，重み,誤差，精度を返して来る．
    ridge_learn = train.Ridge_train(model,inputdata)
    weight_data, acc_data, loss_data, mutial_info_data = ridge_learn.main(weight_in, weight_res, b_in, b_res) 
    sp_acc, tp_acc = acc_data
    sp_loss, tp_loss = loss_data
    loss = (sp_loss + tp_loss)/2
    acc = (sp_acc + tp_acc)/2
    ind.append((acc, loss, weight_data, sp_acc, tp_acc, sp_loss, tp_loss))
  return ind

def next_gene(model,inputdata,ind, pop, weight_in,b_in,children,b_res):
  #第g世代の作成
  for i in range(pop):
    weight_res = children[i]
    #モデルの実行，重み,誤差，精度を返して来る．
    ridge_learn = train.Ridge_train(model,inputdata)
    weight_data, acc_data, loss_data, mutial_info_data = ridge_learn.main(weight_in, weight_res, b_in, b_res)
    sp_acc, tp_acc = acc_data
    sp_loss, tp_loss = loss_data
    loss = (sp_loss + tp_loss)/2
    acc = (sp_acc + tp_acc)/2
    ind.append((acc, loss, weight_data, sp_acc, tp_acc, sp_loss, tp_loss))
  return ind

#選択，生き残る個体を決める関数
def evalution(ind):
  #0で精度、1で誤差，Falseで小さい順，Trueで大きい順
  ind = sorted(ind, key=lambda x:x[1], reverse=False)
  #ind = sorted(ind, key=lambda x:x[0], reverse=True)
  return ind
#二点交叉
def tow_point_crossover(parent1, parent2,gene_length):
  r0 = random.randint(0,31)
  r1 = random.randint(0,gene_length-1)
  r2 = random.randint(r1,gene_length-1)
  child1 = copy.deepcopy(parent1)
  #print(parent1)
  #print(parent2)
  #print(child1)
  child1[r0,r1:r2]= parent2[r0,r1:r2]
  return child1
#突然変異
def mutate(parent,gene_length):
  r1 = random.randint(0, gene_length-1)
  r2 = random.randint(0, gene_length-1)
  child = copy.deepcopy(parent)
  change = random.randint(0,31)
  if child[r1][r2] == change:
    child[r1][r2] = 0
  else:
    child[r1][r2] = change
  return child

def ga_train(args,ind_learn):
  ind = []
  #パラメータ
  gene_length = args.gene_length#切り替わる遺伝子の長さ
  elites = []
  inputdata = input_np.make_data
  name = args.name
  #記録用の変数
  SP_A = []
  TP_A = []
  SP_L = []
  TP_L = []
  W_res = []
  B_res = []
  SP_W = []
  TP_W = []
  G = []
  W = []
  #リザバー以外の重みの固定
  model = ESN_Model.Binde_ESN_Model()
  weight_in, b_in = model.setup_in()
  _, b_res = model.setup_res()
  ana = Analysis.Analysis(name)
  ana.save_in_layer_para(weight_in, b_in)
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
    #ind = []
    rank = 0
    #記録用の処理
    for acc, loss, weight_data,sp_acc, tp_acc, sp_loss, tp_loss in survival:
      #リザバー層重みを取得
      weight_res, b_res, sp_weight_out, tp_weight_out = weight_data
      elites.append(weight_res)
      rank += 1
      print(f'--第{rank}位--空間精度={sp_acc*100:.1f}%-時間精度={tp_acc*100:.1f}%---誤差={loss:.3f}-')
      SP_A.append(sp_acc)
      TP_A.append(tp_acc)
      SP_L.append(sp_loss)
      TP_L.append(tp_loss)
      G.append(g+2)
      W_res.append(weight_res)
      B_res.append(b_res)
      SP_W.append(sp_weight_out)
      TP_W.append(tp_weight_out)
    #記録
    #if G[-1]%10 == 0:
    ana.save_to_ga_data(SP_A,TP_A,SP_L,TP_L,G,W_res,B_res,SP_W,TP_W)
    #次世代に向けた処理
    #変化無しの個体
    #交叉
    while len(elites) < first_pop:
      m1 = random.randint(0,len(elites)-1)#親となる個体の決定
      m2 = random.randint(0,len(elites)-1)#親となる個体の決定
      child = tow_point_crossover(elites[m1],elites[m2],gene_length)#交叉処理
      elites.append(child)
      #突然変異
      if random.random() < args.mutate_rate:
        m = random.randint(0,len(elites)-1)#突然変異する個体を選択
        child = mutate(elites[m],gene_length)
      elites.append(child)
    #次世代の評価
    #args.pop = first_pop-args.survivor
    ind = next_gene(model,inputdata,ind, args.pop, weight_in,b_in, elites,b_res)
    #初期化
    elites = []
  
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  add_arguments(parser)
  args = parser.parse_args()
  print(args)
  #パラメータ
  ind_learn = train.Ridge_train
  ga_train(args,ind_learn)