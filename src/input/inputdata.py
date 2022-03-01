import math
import numpy as np
import random
#dockerでもGUIを使えるようにする
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def makeinput(ntemp=3,tmax=24001):
    temp_period = [8, 16, 32]
    sp_period = np.zeros((3,16))
    sp_period[0] = [-1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1]
    sp_period[1] = [-1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1]
    sp_period[2] = [-1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1]


    #時間と空間の変化がランダムになるように処理
    sp_patt = [np.random.randint(1, ntemp+1) for i in range(tmax)]#24001個に1～3を入れる
    temp_patt = [np.random.randint(1, ntemp+1) for i in range(tmax)]#24001個に1~3を入れる，テストデータを作る時に使う

    #時間パターンのランダムな変化が緩やかになるように処理する
    temp_patt = np.array(temp_patt).T
    for i in range(5):
      temp_patt = np.vstack([temp_patt, temp_patt])
    temp_patt = temp_patt.T
    temp_patt = temp_patt.flatten()

    #空間パターンのランダムな変化が緩やかになるように処理する
    sp_patt = np.array(sp_patt).T
    for i in range(5):
      sp_patt = np.vstack([sp_patt, sp_patt])
    sp_patt = sp_patt.T
    sp_patt = sp_patt.flatten()

    #時間,空間パターンがランダムに切り替わる波の作成
    Itemp = np.zeros((16,tmax))#16*2000の行列
    for i in range(tmax):#24001の長さ
        for j in range(16):
          a = sp_period[(sp_patt[i]-1)]
          Itemp[j,i] = a[j]*(1+(-1+math.cos((2*math.pi)*(i/temp_period[(temp_patt[i]-1)]))))
          #i = a *(1+(-1+cos(2pi)*i/T)
          #i = a cos(2pi*(i/T))

    #正解データの作成
    sp_patt_onehot = np.zeros((tmax,ntemp))#24001*3
    temp_patt_onehot = np.zeros((tmax,ntemp))#24001*3
    for i in range(tmax):
      sp_patt_onehot[i,sp_patt[i]-1] = 1
      temp_patt_onehot[i,temp_patt[i]-1] = 1
    return Itemp, sp_patt_onehot, temp_patt_onehot

if __name__ == '__main__':
  Itemp, sp_patt_onehot, temp_patt_onehot=makeinput(3)
  print(sp_patt_onehot)
  print(Itemp.shape)
  plt.figure()
  plt.imshow(Itemp[:,1000:1400])
  plt.savefig(f'img/input_data.png')