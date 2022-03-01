import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np 
import random
accuracy = []
accuracy2 = []
gene = []
generation = []
num=20#移動平均の個数
b=np.ones(num)/num

read_name = 'ga_ridge_e20_p20_l10/ga_ridge_e20_p20_l10'
read_name = 'ga_ridge2_e20_p20_l10/ga_ridge2_e20_p20_l10'

with open('ga_data/'+read_name+'_sp_acc.csv') as f:
    for row in csv.reader(f):
        accuracy.append(float(row[0])*100)

with open('ga_data/'+read_name+'_tp_acc.csv') as f:
    for row in csv.reader(f):
        accuracy2.append(float(row[0])*100)

for i in range(1000):
  gene.append(i)
for k in range(1000):
  for j in range(10):
    generation.append(k)

acc1 = np.array(accuracy).reshape(-1,10)
ave1 = np.mean(acc1, axis=1)
y_err1 = np.std(acc1, axis=1)
acc2 = np.array(accuracy2).reshape(-1,10)
ave2 = np.mean(acc2, axis=1)
ave2 = np.mean(acc2, axis=1)
y_err2 = np.std(acc2, axis=1)

plt.figure()
fig, ax = plt.subplots()
print(len(accuracy))
print(len(generation))
#print(generation)
print(len(gene))
print(ave1[:len(gene)].shape)
#ax.errorbar(gene,ave1[:len(gene)], yerr=y_err1[:len(gene)],linestyle="None",capsize=3,label="spatial standard deviation",color="lightgreen")
#ax.errorbar(gene,ave2[:len(gene)], yerr=y_err2[:len(gene)],linestyle="None",capsize=3,label="temporal standard deviation",color="lightcoral")
plt.plot(gene,ave1[:len(gene)],label="moving average spatial information",color="g")
plt.plot(gene,ave2[:len(gene)],label="moving average temporal information",color="r")
plt.plot(generation,accuracy,alpha=0.3,label="spatial information",color="g")
plt.plot(generation,accuracy2,alpha=0.3,label="temporal information",color="r")
#plt.yticks((0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0))
plt.yticks((10,20,30,40,50,60,70,80,90,100))
plt.ylim(0,105)
#plt.xlim(-0.5,len(gene)-0.5)
plt.xlim(0,500)
plt.xlabel('Generation',fontsize=15)
plt.ylabel('Accuracy(%)',fontsize=15)
plt.legend(loc=4)
#plt.savefig('img/ga_ridge_1_acc.pdf')
#plt.savefig('img/ga_ridge_1_acc.png')
plt.savefig('img/ga_ridge_e20_p20_l10_acc.png')
plt.savefig('img/ga_ridge_e20_p20_l10_acc.pdf')