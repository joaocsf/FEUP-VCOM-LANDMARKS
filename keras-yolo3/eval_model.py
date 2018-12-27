import os, sys
import xml.etree.ElementTree as ET
from pandas_ml import ConfusionMatrix
import matplotlib.pyplot as plt

def calc_stats(rows):
  totalPred = len(rows)
  
  y_true = []
  y_pred = []
  for i in range(len(rows)):
    row = rows[i]
    className_true = row.split('/')[4]
    numBoxes = int(row.split(' ')[1])
    className_pred = 'none'
    if numBoxes > 0:
      className_pred = row.split(' ')[1+numBoxes].split(',')[4]

    y_true.append(className_true)
    y_pred.append(className_pred)
  
  cm = ConfusionMatrix(y_true, y_pred)
  cm.print_stats()
  cm.stats()

  cm.plot(normalized=True)
  plt.set_cmap('Blues')
  plt.show()

def unionArea(a, b):
  iArea = intersectionArea(a, b)
  sum = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1])
  return sum - iArea

def intersectionArea(a, b):
  dx = min(a[2], b[2]) - max(a[0], b[0])
  dy = min(a[3], b[3]) - max(a[1], b[1])
  if (dx>=0) and (dy>=0):
    return dx*dy
  return 0

def main():
  with open('results.txt') as f:
    lines = f.readlines()
  
  # can be useful later, to evaluate ROI (tier 2)
  # print(intersectionArea([3,3,5,5], [1,1,4,3.5]))
  # print(unionArea([3,3,5,5], [1,1,4,3.5]))
  
  calc_stats(lines)

if __name__ == '__main__':
    main()