import os, sys
import xml.etree.ElementTree as ET
from pandas_ml import ConfusionMatrix
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sn

def calc_general_stats(rows):
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
  
  # stats
  cm = ConfusionMatrix(y_true, y_pred)
  cm.print_stats()
  cm.stats()

  # plot
  cm = confusion_matrix(y_true, y_pred)
  classes = ['arrabida', 'camara', 'clerigos', 'musica', 'none', 'serralves']
  df_cm = pd.DataFrame(cm, index=classes, columns=classes)
  plt.figure(figsize = (10,7))
  sn.set(font_scale=1.4)
  ax = sn.heatmap(cm, annot=True,annot_kws={"size": 16}, yticklabels=classes, xticklabels=classes,cmap='Blues', fmt='g')
  plt.show()

def calc_avg_IoU(rows):  
  col_names =  ['id', 'class', 'IoU']
  df  = pd.DataFrame(columns = col_names)

  for i in range(len(rows)):
    row = rows[i]
    
    numBoxes = int(row.split(' ')[1])
    if numBoxes == 0:
      continue
    className_true = row.split('/')[4]
    className_pred = row.split(' ')[1+numBoxes].split(',')[4]
    if className_pred != className_true:
      continue

    xmin_pred = int(row.split(' ')[1+numBoxes].split(',')[0])
    ymin_pred = int(row.split(' ')[1+numBoxes].split(',')[1])
    xmax_pred = int(row.split(' ')[1+numBoxes].split(',')[2])
    ymax_pred = int(row.split(' ')[1+numBoxes].split(',')[3])
    score = row.split(' ')[1+numBoxes].split(',')[5]
    imageId = row.split('/')[5].split('.')[0]
    
    annotationPath = '../dataset/porto-dataset/annotations/' + className_true + '/' + imageId + '.xml'
    tree = ET.parse(annotationPath)
    root = tree.getroot()
    xmin_true = roundCoords(root.find('object').find('bndbox').find('xmin').text)
    ymin_true = roundCoords(root.find('object').find('bndbox').find('ymin').text)
    xmax_true = roundCoords(root.find('object').find('bndbox').find('xmax').text)
    ymax_true = roundCoords(root.find('object').find('bndbox').find('ymax').text)

    iA = intersectionArea([xmin_pred, ymin_pred, xmax_pred, ymax_pred],[xmin_true, ymin_true, xmax_true, ymax_true])
    uA = unionArea([xmin_pred, ymin_pred, xmax_pred, ymax_pred],[xmin_true, ymin_true, xmax_true, ymax_true])
    IoU = float(iA) / float(uA)

    df.loc[len(df)] = [imageId, className_true, IoU]
  
  means = df.groupby('class').mean()
  print(means)
  print('all', df['IoU'].mean())

def roundCoords(str_):
  return int(float(str_)+0.5)

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

def calc_yolo_scores_stats(rows):
  col_names =  ['id', 'class', 'score', 'isCorrect']
  df  = pd.DataFrame(columns = col_names)

  for i in range(len(rows)):
    row = rows[i]
    
    numBoxes = int(row.split(' ')[1])
    if numBoxes == 0:
      continue
    className_true = row.split('/')[4]
    className_pred = row.split(' ')[1+numBoxes].split(',')[4]
    isCorrect = True
    if className_pred != className_true:
      isCorrect = False

    score = float(row.split(' ')[1+numBoxes].split(',')[5])
    imageId = row.split('/')[5].split('.')[0]
    
    df.loc[len(df)] = [imageId, className_true, score, isCorrect]
  
  means = df.groupby('class').mean()
  print(means)
  means = df.groupby('isCorrect')['score'].mean()
  print(means)
  print('all', df['score'].mean())

def main():
  with open('results.txt') as f:
    lines = f.readlines()
  
  calc_general_stats(lines)
  calc_avg_IoU(lines)
  calc_yolo_scores_stats(lines)

if __name__ == '__main__':
    main()