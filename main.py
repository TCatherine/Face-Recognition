import os
import imageio
from sklearn.model_selection import train_test_split
import numpy as np
import time
import torch
from Model import Model
from Visualization import Show
size_pic=250

def read_data():
    path= os.getcwd() + '/lfw-deepfunneled'
    os.chdir(path)
    datasets=[]
    labels=[]
    for folder in os.listdir():
        datasets_person=[]
        labels.append(folder)
        for image in os.listdir(folder):
            f = open(folder + '/' + image, 'rb')
            img=imageio.imread(f)
            img=np.transpose(img, (2, 0, 1))
            datasets_person.append(img)
        datasets_person=np.array(datasets_person)
        datasets.append(datasets_person)
    datasets=np.array(datasets)
    labels=np.array(labels)
    return datasets, labels

def accuary(model, pair, name, border=65, size_batch =256):
    loss = []
    false_positive=0
    false_negative=0
    total_count=0
    sum_pos=0
    sum_neg=0
    for fnum in range(0, len(pair), size_batch):
        split_pair=pair[fnum:fnum + size_batch]
        temp_loss, distance_of_pos, distance_of_neg=model.Test(split_pair)
        loss.append(temp_loss.detach().numpy())
        sum_pos+=distance_of_pos.sum()
        sum_neg+=distance_of_neg.sum()
        for elem_pos, elem_neg in zip(distance_of_pos, distance_of_neg):
            if elem_pos >= border: false_negative += 1
            if elem_neg < border: false_positive += 1
            total_count+=2
        print('+')
    loss = np.array(loss)
    mean_loss = loss.mean()
    true_previous=total_count-false_negative-false_positive
    print('mean positive: %d; meam negative: %d' % (sum_pos, sum_neg))
    print('%s: Accuracy %d/%d (%.0f%%), False Positive %d, False Negative %d, Loss %.3f;' % (name,
                                                       true_previous, total_count, 100. * true_previous / total_count, false_positive, false_negative,
                                                                                    mean_loss))
    return

def generation(y):
    pairs=[]
    for ind1 in range(len(y)):
        for ind2 in range(len(y[ind1])):
            for ind3 in range(len(y[ind1])):
                if ind3==ind2: continue
                for ind4 in range(len(y)):
                    if ind4==ind1: continue
                    for ind5 in range(len(y[ind4])):
                        num_pair = []
                        num_pair.append(y[ind1][ind2]) #anchor
                        num_pair.append(y[ind1][ind3]) #positive
                        num_pair.append(y[ind4][ind5]) #negative
                        num_pair=np.array((num_pair))
                        pairs.append(num_pair)
    pairs=np.array(pairs)
    return pairs

def train_loop(model, datasets, labels, max_epochs=100):
    pairs = generation(datasets)
    size_batch=256
    train_pairs, test_pairs=train_test_split(pairs, test_size=0.2,
                                          random_state=42, shuffle=False)
    for epoch in range(max_epochs):
        start_time = time.time()
        for felem in range(0, len(train_pairs), size_batch):
            split_pairs=pairs[felem:felem+size_batch]
            model.Train(split_pairs)
        end_time = time.time()
        if epoch % 1 == 0:
            print('\nEpoch %d' % (epoch))
            print('Time of one epoch %.2f' % (end_time - start_time))
            accuary(model, train_pairs, 'Train')
            accuary(model, test_pairs, 'Test')
    Show(model, test_pairs)
    return



if torch.cuda.is_available():
    device = "cuda:0"  # uncomment for running on gpu
    print('GPU')
else:
    device = "cpu"
    print('CPU')
model = Model('Recognizing Face', device)
datasets, labels = read_data()
train_loop(model, datasets, labels)
#ShowExamples(model, test_x, test_labels, 10)
