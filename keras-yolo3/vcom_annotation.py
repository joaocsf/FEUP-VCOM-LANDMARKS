import os, sys
import xml.etree.ElementTree as ET

def roundCoords(str_):
  return str(int(float(str_)+0.5))

def print_it(annotationFilePath, dir_name, files):
  for i in range(len(files)):
    filepath = dir_name + '/' + files[i]
    if os.path.isdir(filepath):
      continue
    
    className = files[i].split('-')[0]
    classId = classes.index(className)

    tree = ET.parse(filepath)
    root = tree.getroot()
    xMin = roundCoords(root.find('object').find('bndbox').find('xmin').text)
    yMin = roundCoords(root.find('object').find('bndbox').find('ymin').text)
    xMax = roundCoords(root.find('object').find('bndbox').find('xmax').text)
    yMax = roundCoords(root.find('object').find('bndbox').find('ymax').text)

    pathToImg = imagesPath + '/' + className + '/' + root.find('filename').text
    if os.path.isfile(pathToImg) == False:
      print('image not found', pathToImg)
      continue

    rowStr = pathToImg + ' ' + xMin + ',' + yMin + ',' + xMax + ',' + yMax + ',' + str(classId) + '\n'
    #print('new row:', rowStr)
    annotationFilePath.write(rowStr)

annotationFilePath = open('train.txt', 'w')
annotationsPath = '../dataset/porto-dataset/annotations'
imagesPath = '../dataset/porto-dataset/images'
classes = ['arrabida', 'camara', 'clerigos', 'musica', 'serralves']
os.path.walk(annotationsPath, print_it, annotationFilePath)