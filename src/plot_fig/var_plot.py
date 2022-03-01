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
pop = 10
point = '/ga_data/ga_ridge_e20_p20_l10/'
name = 'ga_ridge_e20_p20_l10'

with open(point+name+'_h_in_x.csv') as f:
  for row in csv.reader(f):
      h_in_x.append((row[0]))
with open(point+name+'_h_in_y.csv') as f:
  for row in csv.reader(f):
      h_in_y.append((row[0]))
with open(point+name+'_h_out_x.csv') as f:
  for row in csv.reader(f):
      h_out_x.append((row[0]))
with open(point+name+'_h_out_y.csv') as f:
  for row in csv.reader(f):
      h_out_y.append((row[0]))

def change_float(str_list,float_list):
  for i in range(10000):
    num = str_list[i].split(',')
    num[0]=num[0].replace('[','')
    num[-1]=num[-1].replace(']','')
    num = [float(n) for n in num]
    float_list.append(num)

change_float(h_in_x,in_x)
change_float(h_in_y,in_y)
change_float(h_out_x,out_x)
change_float(h_out_y,out_y)

for i in range(pop):



plt.figure()
plt.scatter(in_x,in_y, c='blue',label="input neurons")
plt.scatter(out_x,out_y, c='red',label="output neurons")
#plt.scatter(in_x,in_y, c='blue')
#plt.scatter(out_x,out_y, c='blue')
#plt.xlabel('I_{sp}',fontsize=15)
#plt.ylabel('I_{tp}',fontsize=15)
#plt.legend(loc='upper right')
#plt.legend(fontsize=18)
plt.xlim(0,0.7)
plt.ylim(0,0.7)
plt.savefig('img/mutial_info_conv_cover.png')
plt.savefig('img/mutial_info_conv_cover.svg')
plt.savefig('img/mutial_info_conv_cover.pdf')