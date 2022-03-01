import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

loss = []
top_loss = []
with open('data/acc.csv') as f:
    for row in csv.reader(f):
        loss.append(float(row[0]))

for i in range(len(loss)):
  if i%20 == 0:
    top_loss.append(loss[i])

print(top_loss)

generation =[i+1 for i in range(len(top_loss))]
plt.figure()
#少数表現を禁止
plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
plt.xlabel('Generation')
plt.ylabel('top networks acc')
plt.plot(generation,top_loss[:500])
plt.savefig('img/ga_esn_acc_fix.png')