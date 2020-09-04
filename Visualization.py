
import numpy as np
import matplotlib.pyplot as plt

def create_windows(name, pair, bool, ind, check_bool, num_examples=10):
    # create new windows
    # True Prediction: Similar
    f = plt.figure(figsize=(6, 3*num_examples))
    f.suptitle(name, fontsize=16)
    num_bool = 0
    num_plot=1
    out_pair=[]
    num_prev=-1
    for num in range(1, num_examples + 1):
        while num_bool < len(bool) and bool[num_bool] != check_bool:
            num_bool += 1
        if (num_bool >= len(bool)): break
        if num_prev>-1 and (out_pair[num_prev][ind]==pair[num_bool][ind]).all():
            continue
        out_pair.append(pair[num_bool])
        num_prev+=1
    for pair in out_pair:
        ax = f.add_subplot(len(out_pair),4, num_plot)
        num_plot+=1
        ax.set_xticks([])
        ax.set_yticks([])
        img=np.transpose(pair[0], (1, 2, 0))
        plt.imshow(img)
        ax = f.add_subplot(len(out_pair), 4, num_plot)
        num_plot+=1
        ax.set_xticks([])
        ax.set_yticks([])
        img = np.transpose(pair[ind], (1, 2, 0))
        plt.imshow(img)
        num_bool += 1

def Show(model, pairs, border=65, num_examples=10, size_batch=1024):
    #calculate to truth and false
    for fnum in range(0, len(pairs), size_batch):
        split_pair = pairs[fnum:fnum + size_batch]
        loss, distance_of_pos, distance_of_neg=model.Test(split_pair)
        bool_pos=distance_of_pos<border
        bool_neg=distance_of_neg>=border
    #create new windows
        create_windows('True Prediction: Similar', pairs, bool_pos, 1, True)
        create_windows('False Prediction: Similar', pairs, bool_neg, 2, False)
        create_windows('True Prediction: Dissimilar', pairs, bool_neg, 2, True)
        create_windows('False Prediction: Similar', pairs, bool_pos, 1, False)
        plt.show()

