import numpy as np 
import torch
import torch.nn as nn
from model import binde_esn_model as ESN_Model
import argparse
from PIL import Image
import train as train
from input import input_np
import pickle
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

def add_arguments(parser):
  parser.add_argument('--name', type=str, default='ga_ridge_test', help='save_file_name')
  parser.add_argument('--pop', type=int, default=200, help='pop_model_number')
  parser.add_argument('--survivor', type=int, default=20, help='pop_model_number')
  parser.add_argument('--mutate_rate', default=0.25, help='mutate_rate')
  parser.add_argument('--generation', type=int, default=1000, help='generation')
  parser.add_argument('--gene_length', default=16, help='pop_model_number')
  parser.add_argument('--neuron_num', type=int,default=16, help='use_optimizer')
  parser.add_argument('--write_name', default='atracter', help='savename')
  #python3 src/atracter.py --name 'ga_ridge_4'

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
  #print(len(weight_res2_list))
  weight_res1=weight_res1_list[-20]
  weight_res2=weight_res2_list[-20]
  weight_res3=weight_res3_list[-20]
  weight_res4=weight_res4_list[-20]
  weight_sp_out=weight_sp_out_list[-20]
  weight_tp_out=weight_tp_out_list[-20]
  return weight_res1,weight_res2,weight_res3,weight_res4,weight_sp_out,weight_tp_out

def import_no_serch_weight(args):
  point = 'ga_data/'
  name = args.name
  weight_in= []
  b_in = []
  b_res = []
  with open(f''+point+name+'_bias_weight_in.dat', 'rb') as f:
    weight_in=pickle.load(f)
  with open(f''+point+name+'_bias_in.dat', 'rb') as f:
    bias_in=pickle.load(f)
  with open(f''+point+name+'_bias_res.dat', 'rb') as f:
    bias_res=pickle.load(f)
  return weight_in, bias_in, bias_res

def forward(state_in, state_out, u, weight_in, weight_res1,weight_res2,weight_res3,weight_res4, b_in, b_res,sp_weight_out,tp_weight_out):
  state_in = torch.tanh(torch.matmul(weight_in,u) + b_in + torch.matmul(weight_res1,state_in)+torch.matmul(weight_res2,state_out))
  state_out = torch.matmul(weight_res3,state_in) + torch.matmul(weight_res4,state_out)
  state_in = torch.tanh(state_in)
  state_out = torch.tanh(state_out)
  sp_pre = torch.matmul(sp_weight_out,state_out+ b_res)
  tp_pre = torch.matmul(tp_weight_out,state_out+ b_res)
  return sp_pre, tp_pre, state_in,state_out
  
#??????????????????????????????????????????
def reset_state(size_middle):#16*1
  state = torch.rand((size_middle,1))
  return state

def atracter_calculation(out_data):
  x = []
  y = []
  z = []
  tau = 3
  for i in range(len(out_data)-2*tau):
    x.append(out_data[i])
    y.append(out_data[i+tau])
    z.append(out_data[i+2*tau])
  return x, y, z

def plot_atracter(args,x,y,z,name,sp_test,tp_test):
  x = np.array(x)
  y = np.array(y)
  z = np.array(z)
  print(x.shape)
  for n in range(args.neuron_num):
    fig, ax = rendering(args,x,y,z,sp_test,tp_test,n)
    print(name+'_neuron'+str(n))
    print('-------------plot------------')
    # ????????????
    ax.view_init(elev=15, azim=90)
    ax.set_title("n_"+str(n), fontsize=18)
    #ax.legend(loc='upper left', borderaxespad=0, fontsize=18)
    ax.legend(fontsize=15)
    ax.set_xlabel('X', fontsize=18)
    ax.set_ylabel('Y', fontsize=18)
    ax.set_zlabel('Z', fontsize=18)
    plt.savefig('img/'+args.write_name+'_'+name+'n'+str(n)+'.png')
    print('-------------gif------------')
    # imagemagick???????????????????????????????????????GIF???????????????
    images = [render_frame(ax,angle,fig,name) for angle in range(72)]
    images[0].save('img/'+args.write_name+'_'+name+'n'+str(n)+'.gif', save_all=True, append_images=images[1:], duration=100, loop=0)

def rendering(args,x,y,z,sp_test,tp_test,n):
  fig = plt.figure(figsize=(15,15))
  ax = fig.add_subplot(111, projection='3d')
  time_point = 0
  #print(x.shape[0])
  while int(x.shape[0]) >= time_point:
    #label = 'neuron'+str(i+1)
    if torch.argmax(sp_test[time_point,:]).item() == torch.argmax(sp_test[time_point+29,:]).item() and torch.argmax(sp_test[time_point,:]).item() == torch.argmax(sp_test[time_point+29,:]).item():
      start_time = time_point
      stop_time = start_time+30
      time_point = stop_time
      label = 'sp'+str(torch.argmax(sp_test[start_time,:]).item())+'tp'+str(torch.argmax(tp_test[start_time,:]).item())
    else:
      start_time = time_point
      stop_time = start_time+15
      time_point = stop_time
      label = 'sp'+str(torch.argmax(sp_test[start_time,:]).item())+'tp'+str(torch.argmax(tp_test[start_time,:]).item())
    x_plot = x[start_time:stop_time,n]
    print(x_plot.shape)
    y_plot = y[start_time:stop_time,n]
    z_plot = z[start_time:stop_time,n]
    x_plot = x_plot.flatten()
    y_plot = y_plot.flatten()
    z_plot = z_plot.flatten()
    # axes???????????????????????????
    ax.plot(x_plot, y_plot, z_plot,label=label)
    #ax.scatter(x_plot, y_plot, z_plot,label=label)
  return fig, ax

def render_frame(ax,angle,fig,name):
    """data ??? 3D ???????????? PIL Image ?????????????????????"""
    ax.view_init(30, angle*5)
    plt.close()
    # PIL Image ?????????
    buf = BytesIO()
    fig.savefig(buf, bbox_inches='tight', pad_inches=0.0)
    return Image.open(buf)

if __name__ == '__main__':
  #argpars?????????
  parser = argparse.ArgumentParser()
  add_arguments(parser)
  args = parser.parse_args()
  #???????????????????????????
  inputdata = input_np.make_data
  input, sp_ans, tp_ans = inputdata()
  #???????????????????????????
  model = ESN_Model.Binde_ESN_Model()
  #????????????
  #forward???backward????????????????????????????????????
  #ridge_learn = train.Ridge_train(model,inputdata)
  #???????????????????????????????????????
  weight_res1,weight_res2,weight_res3,weight_res4,weight_sp_out,weight_tp_out = import_serch_weight(args)
  weight_in, b_in, b_res = import_no_serch_weight(args)
  #?????????????????????
  state_in = reset_state(16)
  state_out = reset_state(16)
  #???????????????????????????
  input_atracter = []
  output_atracter = []
  train_point = 1000
  change_point = 1700
  for i in range(train_point,change_point):#12000
    #?????????
    u = input[:,i].reshape(16,1).float()
    sp_pre, tp_pre, state_in, state_out = forward(state_in,state_out,u, weight_in, weight_res1,weight_res2,weight_res3,weight_res4, b_in, b_res,weight_sp_out,weight_tp_out)
    list_state_in = state_in.tolist()
    list_state_out = state_out.tolist()
    #print(np_state_in)
    input_atracter.append(list_state_in)
    output_atracter.append(list_state_out)
  x_1_x,x_1_y,x_1_z = atracter_calculation(input_atracter)
  x_2_x,x_2_y,x_2_z = atracter_calculation(output_atracter)
  #input_neurons
  plot_atracter(args,x_1_x,x_1_y,x_1_z,'x_1',sp_ans,tp_ans)
  #output_neurons
  plot_atracter(args,x_2_x,x_2_y,x_2_z,'x_2',sp_ans,tp_ans)
