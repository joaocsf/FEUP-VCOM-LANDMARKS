import os, sys
import xml.etree.ElementTree as ET

def roundCoords(str_):
  return str(int(float(str_)+0.5))

# 0.2 for test, 0.8 for training
def isForTraining(className, id):
  classId = classes.index(className)
  numImgsOfClass = numImagesByClass[classId]
  id_int = int(id)
  if id_int <= 0.8 * numImgsOfClass:
    return True
  else:
    return False

def print_it(fls, dir_name, files):
  trainFile = fls[0]
  testFile = fls[1]

  for i in range(len(files)):
    filepath = dir_name + '/' + files[i]
    if os.path.isdir(filepath):
      continue
    
    className = files[i].split('-')[0]
    classId = classes.index(className)
    id = files[i].split('-')[1].split('.')[0]

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
    # print('new row:', rowStr)
    if isForTraining(className, id):
      trainFile.write(rowStr)
      # print('train', id)
    else:
      testFile.write(rowStr)
      # print('test', id)

def addNoneImgsToTestFile(trainFile, testFile):
  noneImagesPath = '../dataset/porto-dataset/images/none'
  for subdir, dirs, files in os.walk(noneImagesPath):
    numImgs = len(files)
    id = 0
    for file in files:
      imgPath = os.path.join(subdir, file)
      if id <= 0.8*numImgs:
        trainFile.write(imgPath + ' 0,0,0,0,5\n')
      else:
        testFile.write(imgPath + ' 0,0,0,0,5\n')

      id += 1

trainFile = open('train.txt', 'w')
testFile = open('test.txt', 'w')
annotationsPath = '../dataset/porto-dataset/annotations'
imagesPath = '../dataset/porto-dataset/images'
classes = ['arrabida', 'camara', 'clerigos', 'musica', 'serralves']
numImagesByClass = [521, 452, 575, 325, 205]
for directory, dirnames, filenames in os.walk(annotationsPath):
	for dirname in dirnames:
		dir = os.path.join(directory, dirname)
		files = os.listdir(dir)
		print_it([trainFile, testFile], dir, files)
addNoneImgsToTestFile(trainFile, testFile)