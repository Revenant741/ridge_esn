import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np   
accuracy = []
loss = []
accuracy2 = []
loss2 = []
generation = []

num=20#移動平均の個数
b=np.ones(num)/num

#read_name = 'ga_data/ga_ridge_e20_p20_l10/ga_ridge1_e20_p20_l10'
#read_name = 'ga_data/ga_ridge2_e20_p20_l10/ga_ridge2_e20_p20_l10'
read_name = 'data/p200_l20/ga_ridge/ga_ridge'

with open(read_name+'_sp_acc.csv') as f:
    for row in csv.reader(f):
        accuracy.append(float(row[0])*100)

with open(read_name+'_tp_acc.csv') as f:
    for row in csv.reader(f):
        accuracy2.append(float(row[0])*100)
'''
with open('ga_data/generation.csv') as f:
    for row in csv.reader(f):
        generation.append(float(row[0]))
'''
for i in range(500):
    for j in range(20):
        generation.append(i)
    
accuracy = accuracy[:len(generation)]
accuracy2 = accuracy2[:len(generation)]

conv1 = np.convolve(accuracy, b, mode='same')#移動平均
conv2 = np.convolve(accuracy2, b, mode='same')#移動平均
plt.figure()
plt.gca().yaxis.set_tick_params(which='both', direction='in',bottom=True, top=True, left=True, right=True)
plt.gca().xaxis.set_tick_params(which='both', direction='in',bottom=True, top=True, left=True, right=True)
plt.xlim(0,500)
#plt.yticks((0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0))
plt.yticks((10,20,30,40,50,60,70,80,90,100))
plt.ylim(0,105)
#plt.ylim(0,1)
plt.xlabel('Generation',fontsize=15)
plt.ylabel('Accuracy(%)',fontsize=15)

plt.plot(generation,conv1,label="moving average spatial information",color="g")
plt.plot(generation,conv2,label="moving average temporal information",color="r")
plt.plot(generation,accuracy,alpha=0.3,label="spatial information",color="g")
plt.plot(generation,accuracy2,alpha=0.3,label="temporal information",color="r")
plt.legend(loc=4)
#plt.savefig('img/ga_acc2.png')
plt.savefig('img/ga_ridge_p200_l20_acc.png')
plt.savefig('img/ga_ridge_p200_l20_acc.pdf')


with open(read_name+'_sp_loss.csv') as x:
    for row in csv.reader(x):
        loss.append(float(row[0]))

with open(read_name+'_tp_loss.csv') as x:
    for row in csv.reader(x):
        loss2.append(float(row[0]))

plt.figure()
plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.plot(generation,loss,label="spatial information",color="g")
plt.plot(generation,loss2,label="temporal information",color="r")
plt.legend(loc=0)
plt.savefig('img/ga_ridge_p200_l20_loss.png')
plt.savefig('img/ga_ridge_p200_l20_loss.pdf')
