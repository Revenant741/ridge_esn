import numpy as np 
import torch
import torch.nn as nn
from input import input_np
from model import binde_esn_model as ESN_Model
import train
import argparse
import pickle
import networkx as nx
from networkx.drawing.nx_agraph import to_agraph
from networkx.algorithms import bipartite
import pygraphviz as pgv

def add_arguments(parser):
  parser.add_argument('--name', type=str, default='ga_ridge_test', help='save_file_name')
  parser.add_argument('--pop', type=int, default=200, help='pop_model_number')
  parser.add_argument('--survivor', type=int, default=20, help='pop_model_number')
  parser.add_argument('--mutate_rate', default=0.25, help='mutate_rate')
  parser.add_argument('--generation', type=int, default=1000, help='generation')
  parser.add_argument('--gene_length', default=16, help='pop_model_number')
  parser.add_argument('--neuron_start', type=int, default=0, help='use_optimizer')
  parser.add_argument('--neuron_num', type=int,default=16, help='use_optimizer')
  parser.add_argument('--write_name', default='directed', help='savename')
  parser.add_argument('--conv_num', type=int, default=-20, help='convert_weight_number')
  #python3 src/directed.py --name 'ga_ridge_4' --conv_num -20

def import_serch_weight(args):
  name= args.name
  point = 'ga_data/'
  weight_res1_list = []
  weight_res2_list = []
  weight_res3_list = []
  weight_res4_list = []
  weight_sp_out_list = []
  weight_tp_out_list = []
  with open(f''+point+name+'_weight_res1.dat', 'rb') as f:
    weight_res1_list=pickle.load(f)
  with open(f''+point+name+'_weight_res2.dat', 'rb') as f:
    weight_res2_list=pickle.load(f)
  with open(f''+point+name+'_weight_res3.dat', 'rb') as f:
    weight_res3_list=pickle.load(f)
  with open(f''+point+name+'_weight_res4.dat', 'rb') as f:
    weight_res4_list=pickle.load(f)
  with open(f''+point+name+'_weight_sp_out.dat', 'rb') as f:
    weight_sp_out_list=pickle.load(f)
  with open(f''+point+name+'_weight_tp_out.dat', 'rb') as f:
    weight_tp_out_list=pickle.load(f)
  weight_res1=weight_res1_list[args.conv_num]
  weight_res2=weight_res2_list[args.conv_num]
  weight_res3=weight_res3_list[args.conv_num]
  weight_res4=weight_res4_list[args.conv_num]
  weight_sp_out=weight_sp_out_list[args.conv_num]
  weight_tp_out=weight_tp_out_list[args.conv_num]
  return weight_res1,weight_res2,weight_res3,weight_res4,weight_sp_out,weight_tp_out

def import_no_serch_weight(args):
  point = 'ga_data/'
  name = args.name
  with open(f''+point+name+'_bias_weight_in.dat', 'rb') as f:
    weight_in=pickle.load(f)
  with open(f''+point+name+'_bias_in.dat', 'rb') as f:
    bias_in=pickle.load(f)
  with open(f''+point+name+'_bias_res.dat', 'rb') as f:
    bias_res=pickle.load(f)
  return weight_in, bias_in, bias_res

#エッジ作成関数,重みをノードの色にする場合
def make_edge_color_weight(args,weight,direct,n,k):
  for i in range(args.neuron_start,args.neuron_num):
    for j in range(args.neuron_start,args.neuron_num):
      if 0 < weight[i][j] and weight[i][j] <= 0.4:
        direct.append((i+n,j+k, {"color" : "gold"}))
      elif 0.4 < weight[i][j] and weight[i][j] <= 0.8:
        direct.append((i+n,j+k, {"color" : "darkorange"}))
      elif 0.8 < weight[i][j]:
        direct.append((i+n,j+k, {"color" : "red"}))
  #print(len(direct))
  return direct

#エッジ作成関数,重みをノードの色にする場合
def make_edge(args,weight,direct,n,k):
  for i in range(args.neuron_start,args.neuron_num):
    for j in range(args.neuron_start,args.neuron_num):
      if 0 < weight[i][j] and weight[i][j] <= 0.4:
        direct.append((i+n,j+k,1))
      elif 0.4 < weight[i][j] and weight[i][j] <= 0.8:
        direct.append((i+n,j+k,2))
      elif 0.8 < weight[i][j]:
        direct.append((i+n,j+k,3))
  #print(len(direct))
  #print(direct)
  #print(edge_width)
  return direct



def directed_in(args,in_node,input_node,weight,name):
  direct = []
  #args.neuron_num = 16
  n = -16
  k = 0
  direct = make_edge(args,weight,direct,n,k)
  # 有向グラフの作成
  G = nx.MultiDiGraph()
  G.add_nodes_from(in_node, bipartite=0)
  G.add_nodes_from(input_node, bipartite=1)
  G.add_edges_from(direct)
  nx.write_graphml(G, "test.graphml")
  agraph = nx.nx_agraph.to_agraph(G)
  print('-------------succes------------')
  #ag.node_attr["shape"] = "circle" #  表示方法変更
  agraph.draw('img/'+args.write_name+name+'.png', prog='dot')
  return direct

#Reservoir層の各層の順方向、逆方向の結合
def directed_forward(args,input_node,output_node,weight,name):
  direct = []
  if name == '_input_output':
    n = 0
    k = 16
  else:
    n = 16
    k = 0
  direct = make_edge(args,weight,direct,n,k)
  #print(direct)
  # 有向グラフの作成
  G = nx.MultiDiGraph()
  G.add_nodes_from(input_node, bipartite=1)
  G.add_nodes_from(output_node, bipartite=2)
  G.add_edges_from(direct)
  nx.write_graphml(G, "test.graphml")
  agraph = nx.nx_agraph.to_agraph(G)
  print('-------------succes------------')
  #ag.node_attr["shape"] = "circle" #  表示方法変更
  agraph.draw('img/'+args.write_name+name+'.png', prog='dot')
  return direct

#Reservoir層の各層の内部結合
def directed_return(args,input_node,output_node,weight,name):
  direct = []
  if name == '_output_output':
    n = 16
    k = 16
    direct = make_edge(args,weight,direct,n,k)
    #print(direct)
    # 有向グラフの作成
    G = nx.MultiDiGraph()
    G.add_nodes_from(output_node, bipartite=2)
    G.add_edges_from(direct)
  else:
    n = 0
    k = 0
    direct= make_edge(args,weight,direct,n,k)
    #print(direct)
    # 有向グラフの作成
    G = nx.MultiDiGraph()
    G.add_nodes_from(input_node, bipartite=1)
    G.add_edges_from(direct)
  nx.write_graphml(G, "test.graphml")
  agraph = nx.nx_agraph.to_agraph(G)
  print('-------------succes------------')
  agraph.draw('img/'+args.write_name+name+'.png', prog='dot')
  #ag.node_attr["shape"] = "circle" #  表示方法変更
  return direct


#出力層とOutputNeuronの結合
def directed_out(args,output_node,fc_node,weight,name):
  direct = []
  args.neuron_num = 6
  n = 16
  k = 32
  direct = make_edge(args,weight,direct,n,k)
  # 有向グラフの作成
  G = nx.MultiDiGraph()
  G.add_nodes_from(output_node, bipartite=2)
  G.add_nodes_from(fc_node, bipartite=3)
  G.add_edges_from(direct)
  nx.write_graphml(G, "test.graphml")
  agraph = nx.nx_agraph.to_agraph(G)
  print('-------------succes------------')
  #ag.node_attr["shape"] = "circle" #  表示方法変更
  agraph.draw('img/'+args.write_name+name+'.png', prog='dot')
  return direct

def directed_all(args,direct0,direct1,direct2,direct3,direct4,direct5,in_node,input_node,output_node,fc_node,name):
  # 有向グラフの作成
  G = nx.MultiDiGraph()

  #G.add_edges_from(direct0)
  G.add_edges_from(direct1)
  G.add_edges_from(direct2)
  G.add_edges_from(direct3)
  G.add_edges_from(direct4)
  G.add_edges_from(direct5)

  #G.add_nodes_from(in_node, bipartite=0)
  G.add_nodes_from(input_node, bipartite=1)
  G.add_nodes_from(output_node, bipartite=2)
  G.add_nodes_from(fc_node, bipartite=3)

  nx.write_graphml(G, "test.graphml") 
  agraph = nx.nx_agraph.to_agraph(G)
  print('-------------succes------------')
  agraph.draw('img/'+args.write_name+name+'.png', prog='dot')

if __name__ == '__main__':
  #argparsを設定
  parser = argparse.ArgumentParser()
  add_arguments(parser)
  args = parser.parse_args()

  #モデルをインポート
  model = ESN_Model.Binde_ESN_Model()
  #探索結果の重みをインポート
  weight_res1,weight_res2,weight_res3,weight_res4,weight_sp_out,weight_tp_out = import_serch_weight(args)
  weight_in, b_in, b_res = import_no_serch_weight(args)
  #相互情報量の為の準備
  #forwardとbackwardに当たる部分のインポート
  inputdata = input_np.make_data
  ridge_learn = train.Ridge_train(model,inputdata)
  #リッジ回帰で学習，重み,誤差，精度を返して来る．
  weight_data, acc_data, loss_data, mutial_info_data = ridge_learn.main(weight_in,weight_res1,weight_res2,weight_res3,weight_res4, b_in, b_res)
  h_in_x, h_in_y, h_out_x, h_out_y = mutial_info_data

  #ノードの準備
  input_node = []
  output_node = []
  fc_node = []
  in_node = []
  color_patt = {"color" : "green"}
  color_patt1 = {"color" : "orange"}
  color_patt2 = {"color" : "blue"}
  color_patt3 = {"color" : "red"}
  color_patt4 = {"color" : "yellow"}

  for x in range(-16,0):
    in_node.append((x,color_patt1))

  #Input Neurons
  neuron_num = 0
  for n in range(0,16):
    #print(n)
    if h_in_x[neuron_num] > h_in_y[neuron_num]:
      input_node.append((n,color_patt2))
    elif h_in_x[neuron_num] < h_in_y[neuron_num]:
      input_node.append((n,color_patt3))
    else:
      input_node.append((n,color_patt))
    neuron_num += 1
    

  #Output Neurons
  neuron_num = 0
  for m in range(16,32):
    #print(m)
    if h_out_x[neuron_num] > h_out_y[neuron_num]:
      output_node.append((m,color_patt2))
    elif h_out_x[neuron_num] < h_out_y[neuron_num]:
      output_node.append((m,color_patt3))
    else:
      output_node.append((m,color_patt))
    neuron_num += 1

  for j in range(32,38):
    #print(j)
    fc_node.append((j,color_patt4))
  #print(in_node)
  #print(all_binde)
  #数式的二は　b1*w1+ b2*w21  b3*w12+b4*w2
  weight_fc=torch.cat((weight_sp_out,weight_tp_out),0)
  #print(weight_fc.shape)
  direct0 = directed_in(args,in_node,input_node,weight_in,'_indata')
  direct1 = directed_return(args,input_node,output_node,weight_res1,'_input_input')
  direct2 = directed_return(args,input_node,output_node,weight_res2,'_output_output')
  direct3 = directed_forward(args,input_node,output_node,weight_res3,'_input_output')
  direct4 = directed_forward(args,input_node,output_node,weight_res4,'_output_input')
  direct5 = directed_out(args,output_node,fc_node,weight_fc,'fc')
  directed_all(args,direct0,direct1,direct2,direct3,direct4,direct5,in_node,input_node,output_node,fc_node,'all')
  #direct5 = directed_out(args,output_node,all_binde,fc_node,weight_fc_data,'fc',)