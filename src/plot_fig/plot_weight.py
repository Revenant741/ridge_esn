import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pickle

weight = []

with open('ga_data/bias_weight2.dat', 'rb') as f:
  d2 = pickle.load(f)
  print(d2)
