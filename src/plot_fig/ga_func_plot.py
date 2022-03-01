from builtins import len, print
import csv
from traceback import print_tb
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np 
import random
import seaborn as sns
import pandas as pd
import math

h_in_x = []
h_in_y = []
h_out_x = []
h_out_y = []
in_x = []
in_y = []
out_x = []
out_y = []

#python3 src/plot_fig/ga_func_plot.py

#read_name = 'ga_data/ga_ridge2_e20_p20_l10/ga_ridge2_e20_p20_l10'
read_name = 'ga_data/ga_ridge_e20_p20_l10/ga_ridge_e20_p20_l10'
write_name = 'ga_ridge_p200_l20_func'

with open(read_name+'_h_in_x.csv') as f:
  for row in csv.reader(f):
      h_in_x.append((row[0]))
with open(read_name+'_h_in_y.csv') as f:
  for row in csv.reader(f):
      h_in_y.append((row[0]))
with open(read_name+'_h_out_x.csv') as f:
  for row in csv.reader(f):
      h_out_x.append((row[0]))
with open(read_name+'_h_out_y.csv') as f:
  for row in csv.reader(f):
      h_out_y.append((row[0]))

def change_float(str_list,float_list):
  str_list = np.array(str_list)
  for i in range(len(str_list)):
    num = str_list[i].split(',')
    num[0]=num[0].replace('[','')
    num[-1]=num[-1].replace(']','')
    num = [float(n) for n in num]
    float_list.append(num)

change_float(h_in_x,in_x)
change_float(h_in_y,in_y)
change_float(h_out_x,out_x)
change_float(h_out_y,out_y)
print(len(in_x[0]))
print(in_x[0][1])
print(len(in_x))
#out_x = out_x[-20:]
#out_y = out_y[-20:]
in_x = np.array(in_x)
in_y = np.array(in_y)
out_x = np.array(out_x)
out_y = np.array(out_y)
ga_all_func_diff1 = []
for i in range(len(in_x)):
  sp_func1 =  np.sum(np.abs((in_x[i])-(in_y[i]))/math.sqrt(2))
  tp_func1 =  np.sum(np.abs((out_x[i])-(out_y[i]))/math.sqrt(2))
  eva1= sp_func1 + tp_func1
  ga_all_func_diff1.append(eva1)
  print(f'sp_var_{sp_func1}----tp_var_{tp_func1}----all_var_{eva1}')
print(len(ga_all_func_diff1))

ga_all_func_diff2 = []
for i in range(len(in_x)):
  sp_func2 =  np.sum(math.sqrt(sum((in_x[i])*(in_x[i])+(in_y[i])*(in_y[i]))))
  tp_func2 =  np.sum(math.sqrt(sum((out_x[i])*(out_x[i])+(out_y[i])*(out_y[i]))))
  eva2= sp_func2 + tp_func2
  ga_all_func_diff2.append(eva2)
  print(f'sp_var_{sp_func2}----tp_var_{tp_func2}----all_var_{eva2}')
print(len(ga_all_func_diff2))


generation = int(len(in_x))
gene = [i+1 for i in range(generation)]
#移動平均の個数
ave=20
b=np.ones(ave)/ave
#移動平均の描画
plt.grid()
plt.legend(loc='upper right')
#世代全体で保存する場合
fig = plt.figure()
plt.xlim(0,10000)
plt.ylim(0,4)
plt.xlabel('generated individual number',fontsize=15)
plt.ylabel("$FD1$",fontsize=15)
plt.plot(gene,ga_all_func_diff1,alpha= 1,label="speciality Functional differentiation")
#plt.plot(gene,ga_all_func_diff1,alpha= 1)
move_var=np.convolve(ga_all_func_diff1, b, mode='same')
#plt.plot(gene,move_var,label="average")
plt.legend()
print('-------------succes------------')
#print(ga_all_func_diff1)
print('-------------sort------------')
print(np.argsort(ga_all_func_diff1))
plt.savefig('img/'+write_name+'1.png')
plt.savefig('img/'+write_name+'1.pdf')
plt.grid()
plt.legend(loc='upper right')
#世代全体で保存する場合
fig = plt.figure()
plt.xlim(0,10000)
plt.ylim(0,2.5)
plt.xlabel('generated individual number',fontsize=15)
#plt.ylabel('functional differentiation grade',fontsize=15)
plt.ylabel('$FD2$',fontsize=15)
plt.plot(gene,ga_all_func_diff2,alpha= 1,label="generalizability Functional differentiation")
#plt.plot(gene,ga_all_func_diff2,alpha= 1)
move_var=np.convolve(ga_all_func_diff2, b, mode='same')
#plt.plot(gene,move_var,label="average")
plt.legend()
print('-------------succes------------')
#print(ga_all_func_diff2)
print('-------------sort------------')
print(np.argsort(ga_all_func_diff2))
plt.savefig('img/'+write_name+'2.png')
plt.savefig('img/'+write_name+'2.pdf')
