import cv2 as cv
import numpy as np
import os
import pickle
import re
import time
from pprint import pprint
import argparse
import json

VOCABULARY_SIZE = 50

class MonumentClassifier:

  def __init__(self, database_path, cache_path):
    self.database_path = database_path
    self.annotations_path = os.path.join(database_path, 'annotations/')
    self.images_path = os.path.join(database_path, 'images/')
    self.cache_path=cache_path
    self.vocabulary_path = os.path.join(self.cache_path, 'vocabulary.npy')
    self.__setup_cache_path__()
  
  def __setup_cache_path__(self):
    
    create_dir(self.cache_path)
    self.features_cache_path = os.path.join(self.cache_path, 'features/')
    self.descriptor_cache_path = os.path.join(self.cache_path, 'descriptors/')
    self.svm_cache_path = os.path.join(self.cache_path, 'svm.xml')
    create_dir(self.features_cache_path)
    create_dir(self.descriptor_cache_path)

  def train_vocabulary(self, train_file):
    self.trainfile = train_file
    self.bow_trainer = cv.BOWKMeansTrainer(VOCABULARY_SIZE)

    lines = calculate_lines(self.trainfile)
    current_line = 0
    tms = []
    window = 50
    with open(self.trainfile) as f:
      for line in f:
        current_line += 1
        path, box = line.split(' ')
        safe_name = path_to_cache(path)
        box = [int(x) for x in box.split(',')]
        class_id = box[4]

        start_time=time.time()  

        print(class_id)

        if(class_id == 5):
          box = None
        kp, descriptors, class_id = self.compute_features(path, safe_name, box)

        if len(descriptors) == 0: continue

        self.bow_trainer.add(descriptors)

        end_time=time.time()  
        tms.append(end_time - start_time)
        if(len(tms) > window): tms = tms[-window:]
        mean_time = np.mean(tms)

        lines_left = lines - current_line
        time_left = mean_time * lines_left

        print_bar(current_line/lines, 'VOC', '{0}/{1} ETC: {2}'.format(current_line, lines, time.strftime('%H:%M:%S',time.gmtime(time_left)) ), length=50)

    print('Clustering Phase...', flush=True)
    vocabulary = self.bow_trainer.cluster()
    print('End of Phase...')
    self.store_vocabulary(vocabulary)

  def store_vocabulary(self, vocabulary):
    np.save(self.vocabulary_path, vocabulary)
  
  def load_vocabulary(self):
    return np.load(self.vocabulary_path)
  
  def train_classifier(self, train_file):
    self.trainfile = train_file
    vocabulary = self.load_vocabulary()
    sift = cv.xfeatures2d.SIFT_create()

    flann_params = dict(algorithm=1, trees=5)

    matcher = cv.FlannBasedMatcher(flann_params, {})
    self.bow_extractor = cv.BOWImgDescriptorExtractor(sift, matcher)
    self.bow_extractor.setVocabulary(vocabulary)

    lines = calculate_lines(self.trainfile)
    current_line = 0
    tms = []
    window = 50

    train_data = []
    train_labels = []

    with open(self.trainfile) as f:
      for line in f:
        current_line += 1
        path, box = line.split(' ')
        safe_name = path_to_cache(path)
        box = [int(x) for x in box.split(',')]
        class_id = box[4]

        start_time=time.time()  
        if(class_id == 5):
          box = None
        descriptor = self.compute_descriptor(path, safe_name, box)

        if descriptor is None: continue

        train_data.extend(descriptor)
        train_labels.append(class_id)


        end_time=time.time()  
        tms.append(end_time - start_time)
        if(len(tms) > window): tms = tms[-window:]
        mean_time = np.mean(tms)

        lines_left = lines - current_line
        time_left = mean_time * lines_left

        print_bar(current_line/lines, 'Generating Classification Data', '{0}/{1} ETC: {2}'.format(current_line, lines, time.strftime('%H:%M:%S',time.gmtime(time_left)) ), length=50)
    
    print('Trainning Classifier...', flush=True)
    svm = cv.ml.SVM_create()
    svm.train(np.array(train_data), cv.ml.ROW_SAMPLE, np.array(train_labels))
    svm.trainAuto(np.array(train_data), cv.ml.ROW_SAMPLE, np.array(train_labels), kFold = 30)
    print('Finished Trainning', flush=True)
    self.store_svm(svm)
  
  def test_classifier(self, test_file):
    self.test_file = test_file

    svm = self.load_svm()
    vocabulary = self.load_vocabulary()
    sift = cv.xfeatures2d.SIFT_create()

    flann_params = dict(algorithm=1, trees=5)

    matcher = cv.FlannBasedMatcher(flann_params, {})
    self.bow_extractor = cv.BOWImgDescriptorExtractor(sift, matcher)
    self.bow_extractor.setVocabulary(vocabulary)

    lines = calculate_lines(self.test_file)
    current_line = 0
    tms = []
    window = 50

    predictions = []
    correct = []

    with open(self.test_file) as f:
      for line in f:
        current_line += 1
        path, box = line.split(' ')
        safe_name = path_to_cache(path)
        box = [int(x) for x in box.split(',')]
        class_id = box[4]

        start_time=time.time()  

        if(class_id == 5):
          box = None

        descriptor = self.compute_descriptor(path, safe_name, None)

        if descriptor is None: continue
        #descriptor = descriptor.reshape(1,-1)
        descriptor = np.array(descriptor, dtype=np.float32)
        _, prediction = svm.predict(descriptor)
        prediction = prediction[0][0]
        predictions.append(prediction)
        correct.append(1 if prediction == class_id else 0)
        print(prediction)

        end_time=time.time()  
        tms.append(end_time - start_time)
        if(len(tms) > window): tms = tms[-window:]
        mean_time = np.mean(tms)

        lines_left = lines - current_line
        time_left = mean_time * lines_left

        print_bar(current_line/lines, 'Predicting Data', '{0}/{1} ETC: {2}'.format(current_line, lines, time.strftime('%H:%M:%S',time.gmtime(time_left)) ), length=50)

    result = np.sum(correct) / len(correct)
    print(result)

  def store_svm(self, svm):
    svm.save(self.svm_cache_path)
  
  def load_svm(self):
    return cv.ml.SVM_load(self.svm_cache_path)

  def compute_descriptor(self, image_path, safe_name, box):
    path = os.path.join(self.descriptor_cache_path, safe_name + ".npy")

    if os.path.isfile(path):
      return np.load(path)

    kp, descriptors, class_id = self.compute_features(image_path, safe_name, box)
    descriptor = self.bow_extractor.compute(cv.imread(image_path, cv.IMREAD_GRAYSCALE), kp)
    if len(descriptors) == 0: return None

    np.save(path, descriptor)
    return descriptor

  def compute_features(self, image_path, safe_name, box):
    path = os.path.join(self.features_cache_path, safe_name)
    if(os.path.isfile(path)):
      return load_features(path)
    
    keypoints, descriptors = extract_features(image_path, box)
    class_id = box[4] if not box is None else 5    

    store_features(path, keypoints, descriptors, class_id)
    return (keypoints, descriptors, class_id)


def print_bar(percentage, prefix='Progress...', suffix='Complete', emptychar='-', fullchar='=', length=10):
  size = int(percentage * length)
  bar = fullchar*size + emptychar*(length - size)
  print('{0} |{1}| {2}'.format(prefix, bar, suffix), flush=True)
  

def calculate_lines(filename):
  return sum(1 for line in open(filename))

def path_to_cache(filename):
    return re.sub(r'[\.\/\\]+', '_', filename)

def extract_features(image_path, box):

  image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

  sift = cv.xfeatures2d.SIFT_create()
  kp, des = sift.detectAndCompute(image, None)

  keypoints=[]
  descriptors=[]

  if not box is None:
    xmin = box[0]
    ymin = box[1]
    xmax = box[2]
    ymax = box[3]

    for index, kp in enumerate(kp):
      (x,y) = kp.pt

      if x < xmin or x > xmax or y < ymin or y > ymax: continue
      
      keypoints.append(kp)
      descriptors.append(des[index])
  else:
    keypoints = kp
    descriptors = des

  return (keypoints, np.array(descriptors))

def features_to_pickle(keypoints, descriptors, class_id):
    array = []

    for index, kp in enumerate(keypoints):
        tmp = (
            kp.pt, kp.size,
            kp.angle, kp.response,
            kp.octave, kp.class_id,
            descriptors[index])

        array.append(tmp)
    return [array, class_id]


def pickle_to_features(data):
    keypoints = []
    descriptors = []

    data, class_id = data

    for point in data:
        kp = cv.KeyPoint(
            x=point[0][0], y=point[0][1],
            _size=point[1], _angle=point[2],
            _response=point[3], _octave=point[4],
            _class_id=point[5]
        )
        descriptor = point[6]
        keypoints.append(kp)

        descriptors.append(descriptor)
    return (keypoints, np.array(descriptors), class_id)

def store_features(path, keypoints, descriptors, class_id):
    array = features_to_pickle(keypoints, descriptors, class_id)
    pickle.dump(array, open(path, 'wb'))

def load_features(path):
    array = pickle.load(open(path, 'rb'))
    return pickle_to_features(array)

def create_dir(path):
  if not os.path.isdir(path):
    os.mkdir(path)

def main():
  parser = argparse.ArgumentParser(description='Descriptor Trainer and Classifier')
  parser.add_argument('-train_vocabulary', action='store_true')
  parser.add_argument('-train_classifier', action='store_true')
  parser.add_argument('-test_classifier', action='store_true')
  args = parser.parse_args()
  
  mc = MonumentClassifier('../dataset/porto-dataset/', '../cache/')
  train_file = 'train.txt'
  test_file = 'test.txt'

  if args.train_vocabulary:
    mc.train_vocabulary(train_file)

  if args.train_classifier:
    mc.train_classifier(train_file)
  if args.test_classifier:
    mc.test_classifier(test_file)

if __name__ == "__main__":

  main()