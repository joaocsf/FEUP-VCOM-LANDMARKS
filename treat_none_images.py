import os, sys

noneImagesPath = 'dataset/porto-dataset/images/none'
id = 0
for subdir, dirs, files in os.walk(noneImagesPath):
  for file in files:
    imgPath = os.path.join(subdir, file)
    
    ext = imgPath.split('.')[1]
    if ext != 'jpg':
      os.remove(imgPath)
      continue
    
    newName = 'none-' + str(id).zfill(4) + '.jpg'
    newPath = os.path.join(subdir, newName)
    os.rename(imgPath, newPath)
    
    id += 1