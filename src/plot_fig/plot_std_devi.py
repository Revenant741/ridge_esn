import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np   
accuracy = []
loss = []
accuracy2 = []
loss2 = []
gene = []
survivor = 20 #生き残る個体
num=20#移動平均の個数
b=np.ones(num)/num
read_name = 'bias'

with open('ga_data/'+read_name+'_sp_acc.csv') as f:
    for row in csv.reader(f):
        accuracy.append(float(row[0])*100)

with open('ga_data/'+read_name+'_tp_acc.csv') as f:
    for row in csv.reader(f):
        accuracy2.append(float(row[0])*100)

for i in range(1000):
    for j in range(20):
        gene.append(i)
    
acc1 = np.array(accuracy).reshape(-1,survivor)
ave1 = np.mean(acc1, axis=1)
y_err1 = np.std(acc1, axis=1)
print(acc1.shape)
print(ave1.shape)
acc2 = np.array(accuracy2).reshape(-1,survivor)
ave2 = np.mean(acc2, axis=1)
y_err2 = np.std(acc2, axis=1)

plt.figure()
fig, ax = plt.subplots()
print(len(gene))
print(ave1[:len(gene)].shape)
ax.errorbar(gene,ave1[:len(gene)], yerr=y_err1[:len(gene)],linestyle="None",capsize=5,label="spatial standard deviation",color="lightgreen")
ax.errorbar(gene,ave2[:len(gene)], yerr=y_err2[:len(gene)],linestyle="None",capsize=5,label="temporal standard deviation",color="lightcoral")
plt.plot(gene,ave1[:len(gene)],label="average spatial information",color="g")
plt.plot(gene,ave2[:len(gene)],label="average temporal information",color="r")
plt.xlim(0,len(gene)-1)
#plt.yticks((0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0))
plt.yticks((10,20,30,40,50,60,70,80,90,100))
#plt.xticks((0,1,2,3,4))
plt.ylim(0,105)
plt.xlabel('Generation',fontsize=15)
plt.ylabel('Accuracy(%)',fontsize=15)
plt.legend(loc=4)
read_name='ga_std_devi'
plt.savefig('img/'+read_name+'_acc.svg')
plt.savefig('img/'+read_name+'_acc.png')
plt.savefig('img/'+read_name+'_acc.pdf')