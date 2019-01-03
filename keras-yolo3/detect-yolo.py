import os
from yolo import YOLO
from PIL import Image
import cv2 as cv

def classify(path):
    yolo = YOLO()
    image = Image.open(path)
    r_image = yolo.detect_image(image)
    r_image.show()
    yolo.close_session()

def main():
    yolo = YOLO()
    
    image = Image.open('test-images/musica0.jpg')
    r_image = yolo.detect_image(image)
    r_image.show()
    image = Image.open('test-images/musica-0018.jpg')
    r_image = yolo.detect_image(image)
    cv.imshow('Yolo Result', r_image) 
    r_image.show()
    r_image.save('result.jpg')

    yolo.close_session()


if __name__ == '__main__':
    main()