from hw4_utils import get_pos_and_random_neg, detect, get_iou, generate_result_file, compute_mAP, read_content
from detect import inference_second_stream, prepare_second_stream
import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils import shuffle
from operator import itemgetter
import random
from numpy import linalg as LA
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import tqdm
import os
import cv2


def svm_classifier():
  feat_extractor = prepare_second_stream()
  D_train, lb_train = get_pos_and_random_neg(feat_extractor, 'train',-1)
  model = LinearSVC(max_iter=1000)
  model.fit(D_train, lb_train)
  return model
  #0.00012560011236928403

def negative_mining(loops, neg_count_img, threshold, trainSamples, valSamples, nunegSamples):
  feat_extractor = prepare_second_stream()
  xtrain, ytrain = get_pos_and_random_neg(feat_extractor, 'train',trainSamples)
  xtest, ytest = get_pos_and_random_neg(feat_extractor, 'validation',valSamples)
  model = LinearSVC(max_iter=1000)
  aps = []
  objs = []
  
  for i in range(loops):
    print("Loop"+ str(i))
    model.fit(xtrain, ytrain)
    W = model.coef_
    b = model.intercept_

    yPredProb = model.decision_function(xtest)
    negs_nonsv = []
    negs_sv = []

    obj_sum = 0
    for ind,df in enumerate(model.decision_function(xtrain)):
      if (1-ytrain[ind]*df)<0 and ytrain[ind]==-1:
        negs_nonsv.append(ind)
      if (1-ytrain[ind]*df)>=0:
        obj_sum+= 1-ytrain[ind]*df

    objs.append((LA.norm(W)**2)/2+ obj_sum) 
    #print("xtrain " + str(len(xtrain)))
    #print("negs_nonsv " + str(len(negs_nonsv)))
    xtrain = np.delete(xtrain, negs_nonsv, 0)
    ytrain = np.delete(ytrain, negs_nonsv, 0)
    
    negs_nonsv_len = len(negs_nonsv)
    new_neg_samples = trainSamples*5 - negs_nonsv_len
    negSamples = hard_negative_samples(model, new_neg_samples, neg_count_img, threshold, feat_extractor, nunegSamples)

    #print("negSamples " + str(len(negSamples)))
    xtrain = np.append(xtrain, negSamples, axis=0)
    ytrain = np.append(ytrain, -1 * np.ones(len(negSamples)), axis = 0)


    xtrain, ytrain = shuffle(xtrain, ytrain, random_state=0)


    #print("Average Precision Score ", average_precision_score(ytest, yPredProb))
    aps.append(average_precision_score(ytest, yPredProb))

  return model, objs, aps


def hard_negative_samples(svm_model, neg_count, neg_count_img, threshold, feat_extractor, nunegSamples):
  images_pool = "ContactHands/JPEGImages/"
  dataset_file = "ContactHands/ImageSets/Main/validation.txt"
  annotations_pool = "ContactHands/Annotations/"
  model2 = prepare_second_stream()
  negCount = 0

  image_size = (256, 256)
  with open(dataset_file, "r") as f:
      dataset = f.read().splitlines()
  dataset = random.sample(dataset, nunegSamples)
  featsmaster = []

  for img_name in tqdm.tqdm(dataset):
    rects = []
    img_path = os.path.join(images_pool, img_name + ".jpg")
    xml_path = os.path.join(annotations_pool, img_name + ".xml")
    image_file, boxes = read_content(xml_path)

    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        img = cv2.resize(img, image_size)

        rects = detect(img, feat_extractor, svm_model, -1)

        rects = rects[:, :-1]
        rectnegMaster = []

        maxCount = neg_count_img
        for rect in rects:
          negSample = len([iou for iou in get_iou(rect, np.array(boxes)) if iou > threshold])
          if negSample==0:
            rectnegMaster.append(rect)
            maxCount-=1
          if maxCount==0:
            break

        feats = feat_extractor.extract_features(rectnegMaster)
        feats = feats.detach().to('cpu').numpy()

        for feat in feats:
          featsmaster.append(feat)
        negCount += len(rectnegMaster)

    if negCount>=neg_count:
      break

  return np.array(featsmaster)