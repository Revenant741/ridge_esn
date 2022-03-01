from builtins import len, print
import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np 
import random
import seaborn as sns
import pandas as pd

h_in_x = []
h_in_y = []
h_out_x = []
h_out_y = []
in_x = []
in_y = []
out_x = []
out_y = []
#python3 src/plot_fig/mutial_info_plot.py
#read_name = 'ga_data/ga_ridge2_e20_p20_l10/ga_ridge2_e20_p20_l10'
#read_name = 'ga_data/ga_ridge_e20_p20_l10/ga_ridge_e20_p20_l10'
read_name = 'data/p200_l20/ga_ridge/ga_ridge'
write_name = 'ga_ridge_p200_l20'

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
  for i in range(20):
    num = str_list[i].split(',')
    num[0]=num[0].replace('[','')
    num[-1]=num[-1].replace(']','')
    num = [float(n) for n in num]
    float_list.append(num)

change_float(h_in_x,in_x)
change_float(h_in_y,in_y)
change_float(h_out_x,out_x)
change_float(h_out_y,out_y)

print(len(in_x))
in_x = in_x[:10]
in_y = in_y[:10]
out_x = out_x[:10]
out_y = out_y[:10]
print(len(in_x))
print(len(in_y))
print(len(out_x))
print(len(out_y))
#print(in_x)

plt.figure()
plt.scatter(in_x,in_y, c='blue',label="input neurons")
plt.scatter(out_x,out_y, c='red',label="output neurons")
#plt.scatter(in_x,in_y, c='blue')
#plt.scatter(out_x,out_y, c='red')
plt.xlabel('$I(h_{i}~;~I_{sp})$',fontsize=15)
plt.ylabel('$I(h_{i}~;~I_{tp})$',fontsize=15)
plt.legend(loc='upper right')
plt.legend(fontsize=18)
plt.xlim(0,0.7)
plt.ylim(0,0.7)
plt.savefig('img/'+write_name+'_mi.png')
plt.savefig('img/'+write_name+'_mi.pdf')
#plt.savefig('img/ga_ridge_e20_p20_l10_mi.svg')
#plt.savefig('img/ga_ridge_e20_p20_l10_mi.pdf')
