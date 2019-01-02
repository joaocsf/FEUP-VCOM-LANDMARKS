import os
from yolo import YOLO
from PIL import Image

def main():
  resultsFile = open('results.txt', 'w')
  with open('test_yolo.txt') as f:
    lines = f.readlines()
  
  yolo = YOLO()  

  for i in range(len(lines)):
    row = lines[i]
    rowSplited = row.split(' ')
    imgPath = rowSplited[0]
    print(i, '/', len(lines))
    imgPath = imgPath.replace('\n','')
    image = Image.open(imgPath)
    r_image = yolo.vcom_detect_image(image, imgPath, resultsFile)

  yolo.close_session()

if __name__ == '__main__':
    main()