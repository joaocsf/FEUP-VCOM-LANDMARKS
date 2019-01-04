import cv2 as cv
import numpy as np
import os
import pickle
import re
import time
from pprint import pprint
import argparse
import json
import multiprocessing.dummy as mp
import random

VOCABULARY_SIZE = 100
MIN_MATCH_COUNT = 20

class MonumentClassifier:

  ## Constructor wich setups all the paths needed
  def __init__(self, database_path, cache_path):
    self.database_path = database_path
    self.annotations_path = os.path.join(database_path, 'annotations/')
    self.images_path = os.path.join(database_path, 'images/')
    self.cache_path=cache_path
    self.DEBUG=False
    self.vocabulary_path = os.path.join(self.cache_path, 'vocabulary.npy')
    self.__setup_cache_path__()
    self.multi_thread = False

  # Method to set the multi_thread variable
  def set_multi_thread(self, active):
    self.multi_thread = active

  # Helper Method to setup the cache directory
  def __setup_cache_path__(self):
    
    create_dir(self.cache_path)
    self.features_cache_path = os.path.join(self.cache_path, 'features/')
    self.descriptor_cache_path = os.path.join(self.cache_path, 'descriptors/')
    self.svm_cache_path = os.path.join(self.cache_path, 'svm.xml')
    self.lookup_cache = os.path.join(self.cache_path, 'lookup.json')
    create_dir(self.features_cache_path)
    create_dir(self.descriptor_cache_path)

  # Method to process the vocabulary per line
  def process_vocabulary(self, line):
    print('Processing...', line)
    self.counter +=1
    path, safe_name, box, class_id = parseline(line)

    if(class_id == 5):
      box = None

    #Compute the descriptors for the image
    kp, descriptors, class_id = self.compute_features(path, safe_name, None)

    if len(descriptors) == 0: return

    #Add the descriptors to the bow trainer
    self.bow_trainer.add(descriptors)
    #Print progress
    print_bar(self.counter/self.max_lines, 'VOC', '{0}/{1}'.format(self.counter, self.max_lines), length=50)

  #Method to train the vocabulary using the BOWKMeansTrainer
  def train_vocabulary(self, train_file):
    self.trainfile = train_file
    self.bow_trainer = cv.BOWKMeansTrainer(VOCABULARY_SIZE)
    self.counter = 0

    lines = calculate_lines(self.trainfile)
    self.max_lines = lines
    self.counter = 0
    current_line = 0
    tms = []
    window = 50

    with open(self.trainfile) as f:
      if self.multi_thread:
        p = mp.Pool(8)
        print('Multithreading')
        p.map(self.process_vocabulary, f.readlines())
        p.close()
        p.join()

      else:
        for line in f:
          self.process_vocabulary(line)

    print('Clustering Phase...', flush=True)
    vocabulary = self.bow_trainer.cluster()
    print('End of Phase...')
    self.store_vocabulary(vocabulary)

  # Method to store the vocabulary for future usage
  def store_vocabulary(self, vocabulary):
    np.save(self.vocabulary_path, vocabulary)
  
  # Method to reload the vocabulary
  def load_vocabulary(self):
    return np.load(self.vocabulary_path)
  
  # Method to process a line to train the classifier
  def process_train_classifier(self, line):
    print('Processing...', line)
    self.counter +=1
    path, safe_name, box, class_id = parseline(line)

    if(class_id == 5):
      box = None

    # Compute the BoW descriptor
    descriptor = self.compute_descriptor(path, safe_name, None)

    if descriptor is None: return

    # Store the BoW descriptor and label value
    self.train_data.extend(descriptor)
    self.train_labels.append(class_id)

    #Print progress
    print_bar(self.counter/self.max_lines, 'Train', '{0}/{1}'.format(self.counter, self.max_lines), length=50)

  #Method to train the classifier using single or multithread (Uses SIFT and BOWImgDescriptorExtractor)
  def train_classifier(self, train_file):
    self.trainfile = train_file
    self.counter = 0
    vocabulary = self.load_vocabulary()
    sift = cv.xfeatures2d.SIFT_create()

    flann_params = dict(algorithm=1, trees=5)

    matcher = cv.FlannBasedMatcher(flann_params, {})
    self.bow_extractor = cv.BOWImgDescriptorExtractor(sift, matcher)
    self.bow_extractor.setVocabulary(vocabulary)

    lines = calculate_lines(self.trainfile)
    self.max_lines = lines
    current_line = 0
    tms = []
    window = 50

    self.train_data = []
    self.train_labels = []

    with open(self.trainfile) as f:
      if self.multi_thread:
        p = mp.Pool(8)
        print('Multithreading')
        p.map(self.process_train_classifier, f.readlines())
        p.close()
        p.join()

      for line in f:
        self.process_train_classifier(line)
    
    print('Trainning Classifier...', flush=True)
    svm = cv.ml.SVM_create()
    #svm.train(np.array(train_data), cv.ml.ROW_SAMPLE, np.array(train_labels))
    svm.trainAuto(np.array(self.train_data), cv.ml.ROW_SAMPLE, np.array(self.train_labels), kFold = 50)
    print('Finished Trainning', flush=True)
    self.store_svm(svm)

  # Process to parse a line from the train set and categorize by the label/class id
  def process_divide_lookup(self, line):
    print('Processing...', line)
    self.counter +=1
    path, safe_name, box, class_id = parseline(line)

    if not self.lookup_data.__contains__(class_id):
      self.lookup_data[class_id] = []
    
    self.lookup_data[class_id].append(line)

    print_bar(self.counter/self.max_lines, 'Lookup', '{0}/{1}'.format(self.counter, self.max_lines), length=50)
  

  # Process to create a lookup dictionary for latter user 
  def divide_lookup(self, train_file):
    self.lookup_data = {}
    self.counter = 0
    self.max_lines = calculate_lines(train_file)

    with open(train_file) as f:
      if self.multi_thread:
        p = mp.Pool(8)
        print('Multithreading')
        p.map(self.process_divide_lookup, f.readlines())
        p.close()
        p.join()
      else:
        for line in f:
          self.process_divide_lookup(line)

    self.store_lookup(self.lookup_cache, self.lookup_data)

  # Method to store the lookup dictionary
  def store_lookup(self, path, lookup_data):
    with open(path, 'w') as out:
      json.dump(lookup_data, out)
  
  # Method to load the lookup dictionary
  def load_lookup(self, path):
    with open(path) as input:
      return json.load(input)

  # Method to classify a single image
  def classify(self, path):
    self.lookup_data = self.load_lookup(self.lookup_cache)
    self.counter = 0
    self.results = []
    self.svm = self.load_svm()

    vocabulary = self.load_vocabulary()
    sift = cv.xfeatures2d.SIFT_create()
    flann_params = dict(algorithm=1, trees=5)
    matcher = cv.FlannBasedMatcher(flann_params, {})
    self.bow_extractor = cv.BOWImgDescriptorExtractor(sift, matcher)
    self.bow_extractor.setVocabulary(vocabulary)
    self.max_lines = 1

    self.predictions = []
    self.correct = []

    self.process_test_classifier('{0} 0,0,0,0,0'.format(path))

  # Method to classify a single line
  def process_test_classifier(self, line):
    print('Processing...', line)
    self.counter +=1
    path, safe_name, box, class_id = parseline(line)

    if(class_id == 5):
      box = None

    # Compute the BoW Descriptor
    descriptor = self.compute_descriptor(path, safe_name, None)

    if descriptor is None: return
    descriptor = np.array(descriptor, dtype=np.float32)
    _, prediction = self.svm.predict(descriptor)
    prediction = prediction[0][0]
    self.predictions.append(prediction)

    #Attempt up to 40 times to match the image with the trainning set using the lookup dictionary stored previously
    tries = 40
    box = None
    if prediction != 5:
      for i in range(tries):
        rnd = random.choice(self.lookup_data[str(int(prediction))])
        #rnd = self.lookup_data[str(int(prediction))][0]
        print(rnd)
        box = self.compute_box(rnd, line)

        if not box is None: break

    #FAILURE
    if box is None:
      #print('\nFAILURE{0}\n'.format(class_id), flush=True)
      prediction = 5
    else:
      img = cv.imread(path, cv.IMREAD_COLOR)
      box = [int(x) for x in box]

    self.save_to_resuts(path, box, prediction)
    self.correct.append(1 if prediction == class_id else 0)
    print_bar(self.counter/self.max_lines, 'Test', '{0}/{1}'.format(self.counter, self.max_lines), length=50)

  # Method to compute the bounding box given a image from the lookup_line
  def compute_box(self, lookup_line, test_line):
    l_path, l_name, l_box, l_class = parseline(lookup_line)
    t_path, t_name, t_box, t_class = parseline(test_line)

    # Retrieve the resized image
    l_image , l_factor = open_resized_image(l_path)
    t_image , t_factor = open_resized_image(t_path)

    sift = cv.xfeatures2d.SIFT_create()

    # Retrive the resize factor and adjust the coordinates
    m_box = [int(x*l_factor) for x in l_box]
    l_kp, l_des = extract_features(l_path, m_box, l_image)
    t_kp, t_des = sift.detectAndCompute(t_image, None)

    l_color_image = cv.cvtColor(l_image, cv.COLOR_GRAY2BGR)
    t_color_image = cv.cvtColor(t_image, cv.COLOR_GRAY2BGR)

    # Draw the original bounding box on the resized image
    pt1 = [m_box[0], m_box[1]]
    pt2 = [m_box[2], m_box[1]]
    pt3 = [m_box[2], m_box[3]]
    pt4 = [m_box[0], m_box[3]]
    pts = np.array([pt1, pt2, pt3, pt4])
    l_color_image = cv.drawContours(l_color_image, [pts], -1, (255,0,0), 4)
    # Draw the keypoints
    cv.drawKeypoints(l_color_image, l_kp, l_color_image)
    cv.drawKeypoints(t_color_image, t_kp, t_color_image)
    #cv.namedWindow('l_image', cv.WINDOW_NORMAL)
    #cv.namedWindow('t_image', cv.WINDOW_NORMAL)
    #cv.imshow('l_image', l_color_image)
    #cv.imshow('t_image', t_color_image)
    #cv.waitKey(0)

    # Initialize the bruteforce matcher
    feature_matcher = cv.BFMatcher()
    matches = feature_matcher.match(l_des, t_des)

    # Sort matches by distance
    matches = sorted(matches, key = lambda x: x.distance)
    if matches is None: return None

    tmp = []
    h, w = l_image.shape[:2]
    thresh = max(h, w)

    # Attempt to limit the number of matches
    for m in matches:
      if(m.distance < 512):
        tmp.append(m)

    # Check if enought matches exist
    matches = tmp
    if len(matches) > MIN_MATCH_COUNT:
      good = matches
      pts1 = np.float32(
            [l_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
      pts2 = np.float32(
            [t_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
      
      # Calculate the homography
      H, mask = cv.findHomography(pts1, pts2, cv.RANSAC, 5.0)

      # Calculate the number of inliners used to compute the homography
      mask_list = mask.ravel().tolist()
      n_inliners = np.sum(mask_list)

      if n_inliners < MIN_MATCH_COUNT:
        return None
      
      if H is None:
        return None

      # translate the keypoints from the resized image to the real scale of the second's image
      l_box_points = ((l_box[0], l_box[1]), (l_box[2], l_box[3])) 
      #r_box_points = [ mult_point(mult(H, mult_point(x, l_factor)), 1/t_factor) for x in l_box_points]
      r_box_points = [ mult(H, mult_point(x, l_factor)) for x in l_box_points]
      #r_box_points = [ mult(H, x) for x in l_box_points]

      x_dist = r_box_points[0][0] - r_box_points[1][0]
      y_dist = r_box_points[0][1] - r_box_points[1][1]

      tmp_area = abs(x_dist * y_dist)

      h,w = t_image.shape[:2]
      #Filter the bounding box area
      if tmp_area < (h*w)/128:
        return None

      #Debugging Code block to draw the image matches and the resulting bounding box
      if self.DEBUG:
        tmp_pts = [mult(H, mult_point(x, l_factor)) for x in l_box_points]
        p1, p2 = tmp_pts
        p1 = [int(x) for x in p1]
        p2 = [int(x) for x in p2]
        tmp_pts = ((p1[0], p1[1]), (p2[0], p1[1]), (p2[0], p2[1]), (p1[0], p2[1]))
        tmp_pts = np.array(tmp_pts)
        t_color_image = cv.drawContours(t_color_image, [tmp_pts], -1, (0,0,255), 4)

        draw_params = dict(matchColor = (0,255,0),
                      singlePointColor = None,
                      matchesMask = mask_list,
                      flags = 2)
        img = cv.drawMatches(l_color_image, l_kp, t_color_image, t_kp, good, None, **draw_params)
        cv.namedWindow('BoW Matches', cv.WINDOW_NORMAL)
        cv.imshow('BoW Matches', img)
        cv.waitKey(1)

      r_box_points = [clamp_point(x, (0,0), (w,h)) for x in r_box_points]

      r_box_points = [mult_point(x, 1/t_factor) for x in r_box_points]


      x_a, y_a = r_box_points[0]
      x_b, y_b = r_box_points[1]

      r_box = (min(x_a, x_b), min(y_a, y_b), max(x_a, x_b), max(y_a, y_b))
      return r_box
    else:
      return None

  # Method to store the results to a friendly data structure
  def save_to_resuts(self, image_path, box, classification):
    self.results.append((image_path, (box, classification)))

  # Method to parse the results data structure and save asn a general results file
  def store_results(self):
    names = ['arrabida','camara','clerigos','musica','serralves', 'None']
    with open('results.txt', 'w') as f:
      for line in self.results:
        image_path, prediction = line
        box, classification = prediction
        prediction_string = "0"


        if not box is None:
          box = [ 0 if x < 0 else x for x in box]
          prediction_string = "1 {0},{1},{2},{3},{4},1.0".format(box[0], box[1], box[2], box[3], names[int(classification)])

        f.write("{0} {1}\n".format(image_path, prediction_string))


  # Method to test the classifier (setups the svm classifier and BoW Descriptor Extractor)
  def test_classifier(self, test_file):
    self.lookup_data = self.load_lookup(self.lookup_cache)
    self.test_file = test_file
    self.counter = 0
    self.results = []
    self.svm = self.load_svm()
    vocabulary = self.load_vocabulary()
    sift = cv.xfeatures2d.SIFT_create()

    flann_params = dict(algorithm=1, trees=5)

    matcher = cv.FlannBasedMatcher(flann_params, {})
    self.bow_extractor = cv.BOWImgDescriptorExtractor(sift, matcher)
    self.bow_extractor.setVocabulary(vocabulary)

    lines = calculate_lines(self.test_file)
    self.max_lines = lines
    current_line = 0
    tms = []
    window = 50

    self.predictions = []
    self.correct = []

    with open(self.test_file) as f:
      if self.multi_thread:
        p = mp.Pool(8)
        print('Multithreading')
        p.map(self.process_test_classifier, f.readlines())
        p.close()
        p.join()

      for line in f:
        self.process_test_classifier(line)

    result = np.sum(self.correct) / len(self.correct)
    print('Saving Results, Estimated Accuracy {0}'.format(result))
    self.store_results()

  # Method to store the SVM classifier
  def store_svm(self, svm):
    svm.save(self.svm_cache_path)
  
  # Method to load the SVM classifier
  def load_svm(self):
    return cv.ml.SVM_load(self.svm_cache_path)

  # Method to retrieve the BoW descriptor (caching it for latter use)
  def compute_descriptor(self, image_path, safe_name, box):
    path = os.path.join(self.descriptor_cache_path, safe_name + ".npy")

    if os.path.isfile(path):
      return np.load(path)

    kp, descriptors, class_id = self.compute_features(image_path, safe_name, box)
    descriptor = self.bow_extractor.compute(open_image(image_path), kp)
    if len(descriptors) == 0: return None

    np.save(path, descriptor)
    return descriptor

  # Method to retrieve the image features (caching it for latter use)
  def compute_features(self, image_path, safe_name, box):
    path = os.path.join(self.features_cache_path, safe_name)
    if(os.path.isfile(path)):
      return load_features(path)
    
    keypoints, descriptors = extract_features(image_path, box)
    class_id = box[4] if not box is None else 5    

    store_features(path, keypoints, descriptors, class_id)
    return (keypoints, descriptors, class_id)


# Helper method to multiply an homography by an homogeneous point
def mult(H, p):
  rp = H @ (p[0], p[1], 1)
  dp = (rp[0] / rp[2], rp[1] / rp[2])
  return dp

# Helper method to print a progress bar
def print_bar(percentage, prefix='Progress...', suffix='Complete', emptychar='-', fullchar='=', length=10):
  size = int(percentage * length)
  bar = fullchar*size + emptychar*(length - size)
  print('{0} |{1}| {2}'.format(prefix, bar, suffix), flush=True)

# Helper method open an image
def open_image(filename):
  image, _ = open_resized_image(filename)
  return image

# Helper method open an resized image up to 512x512
def open_resized_image(filename):
  image = cv.imread(filename, cv.IMREAD_GRAYSCALE)
  height, width = image.shape[:2]
  n_width = 512
  r = n_width/width
  n_height = int(height * r)
  return cv.resize(image, (n_width, n_height)) , r

# Helper method parse a line from the testing or trainning set
def parseline(line):
  path, box = line.split(' ')
  safe_name = path_to_cache(path)
  box = [int(x) for x in box.split(',')]
  class_id = box[4]
  return (path, safe_name, box, class_id)

#Method to retrieve the number of lines of a file
def calculate_lines(filename):
  return sum(1 for line in open(filename))

#Method to transform a filename to a safe to store file-name
def path_to_cache(filename):
    return re.sub(r'[\.\/\\:]+', '_', filename)

#Method to extract the features from an image, with option to limit by the bounding box
def extract_features(image_path, box, image=None):

  image = image if not image is None else open_image(image_path)

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

#Method to store the features to a pickle structure
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


#Method to transform a pickle structure to feature structure
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

#Method to multiply a point by a value
def mult_point(point, value):
  return (point[0] * value, point[1] * value)

#Method to store features
def store_features(path, keypoints, descriptors, class_id):
    array = features_to_pickle(keypoints, descriptors, class_id)
    pickle.dump(array, open(path, 'wb'))

#Method to load features
def load_features(path):
    array = pickle.load(open(path, 'rb'))
    return pickle_to_features(array)

#Method to create a directory if it doesn't exists
def create_dir(path):
  if not os.path.isdir(path):
    os.mkdir(path)

#Method to clamp a point
def clamp_point(p, minimum, maximum):
  return (clamp(p[0], minimum[0], maximum[0]),
      clamp(p[1], minimum[1], maximum[1]))

#Method to clamp a value
def clamp(p, minimum, maximum):
  if p < minimum:
    return minimum
  if p > maximum:
    return maximum
  return p

#Method to classify a single image
def classify(image_path, root):
  mc = MonumentClassifier('../dataset/porto-dataset/'.format(root), '../cache/'.format(root))
  mc.DEBUG = True
  mc.classify(image_path)

#Main method to setup de programs arguments and the MonumentClassifier class
def main():
  parser = argparse.ArgumentParser(description='Descriptor Trainer and Classifier')
  parser.add_argument('-train_vocabulary', action='store_true')
  parser.add_argument('-train_classifier', action='store_true')
  parser.add_argument('-test_classifier', action='store_true')
  parser.add_argument('-build_lookup', action='store_true')
  args = parser.parse_args()
  
  mc = MonumentClassifier('../dataset/porto-dataset/', '../cache/')
  train_file = 'train.txt'
  test_file = 'test.txt'

  mc.set_multi_thread(True)

  if args.train_vocabulary:
    mc.train_vocabulary(train_file)

  if args.train_classifier:
    mc.train_classifier(train_file)

  if args.build_lookup:
    mc.divide_lookup(train_file)
    
  if args.test_classifier:
    mc.test_classifier(test_file)

if __name__ == "__main__":
  main()