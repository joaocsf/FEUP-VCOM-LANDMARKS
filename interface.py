import tkinter as tk
import tkinter.filedialog
import sys
import os
from features.features import classify as classify_bow
from PIL import Image
from pprint import pprint
import importlib
import cv2 as cv

sys.path.append('./keras-yolo3/')


YOLO = importlib.import_module('keras-yolo3.yolo')

os.chdir('keras-yolo3/')
yolo = YOLO.YOLO()
os.chdir('../')

use_yolo = False
algorithm_button = None

def classify_yolo(path):
  global yolo
  image = Image.open(path)
  r_image = yolo.detect_image(image)
  r_image.show()

def toggle_yolo():
  global algorithm_button, use_yolo
  use_yolo = not use_yolo
  algorithm_button.config(text="Yolo" if use_yolo else "BoW")

def open_image():
  filePath = tk.filedialog.askopenfilename(
      title='Select a Map\'s Image',
      filetypes=(
          ('all files', '*'),
          ('jpeg files', '*.jpg'),
          ('png files', '*.png')
      )
  )
  classify_image(filePath)

def classify_image(path):
  global use_yolo

  if use_yolo:
    os.chdir('./keras-yolo3/')
    try:
      classify_yolo(path)
    except:
      pass
    os.chdir('../')
  else:
    os.chdir('./features/')
    classify_bow(path, '.')
    os.chdir('../')

  print('Finished', flush=True)

def main():
  global use_yolo, algorithm_button, yolo
  top = tk.Tk()
  algorithm_button = tk.Button(top)
  algorithm_button['command'] = toggle_yolo
  algorithm_button.grid(column=0, row=0)

  img = tk.Button(top)
  img['text']= "Open Image"
  img['command'] = open_image
  img.grid(column=1, row=0)

  toggle_yolo()
  top.mainloop()
  yolo.close_session()


if __name__ == "__main__":
  main()