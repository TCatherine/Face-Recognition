import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

def conv_block(in_f, out_f, ker, stride, padding=1):
    return nn.Sequential(nn.Conv2d(in_f, out_f, ker, stride, padding),
                         nn.MaxPool2d(ker, stride),
                         nn.BatchNorm2d(out_f),
                         nn.LeakyReLU())

class Net(nn.Module):
    # @property
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = conv_block(3, 32, 4, 2, 1)
        self.conv2=conv_block(32, 64, 4, 2, 1)

        self.linear1=nn.Linear(12544, 128)
        self.fun3=nn.ReLU()
        #self.linear2=nn.Linear(128, 2)

        self.sigmoid=nn.Sigmoid()

    def forward(self, data):
        con1=self.conv1(data)
        con2=self.conv2(con1)
        var=con2.reshape(con2.shape[0], -1)
        variable = self.linear1(var)
        variable=self.fun3(variable)

        return variable

class Model:
    def __init__(self, name, device):
        self.name = name
        self.net = Net()
        #self.optimizer = torch.optim.SGD(self.net.parameters(), lr=0.1)
        self.triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.1)
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.device = torch.device(device)
        if device != "cpu":
            self.net.cuda(self.device)

    def function(self, pairs):
        pair = np.array(pairs)
        pair = np.transpose(pair, (1, 0, 2, 3, 4))
        if (torch.cuda.is_available()):
            a = torch.cuda.FloatTensor(pair[0] / 255, device=self.device)
            p = torch.cuda.FloatTensor(pair[1] / 255, device=self.device)
            n = torch.cuda.FloatTensor(pair[2] / 255, device=self.device)
        else:
            a = torch.FloatTensor(pair[0] / 255, device=self.device)
            p = torch.FloatTensor(pair[1] / 255, device=self.device)
            n = torch.FloatTensor(pair[2] / 255, device=self.device)
        fun_anchor = self.net(a)
        fun_positive = self.net(p)
        fun_negative = self.net(n)
        return fun_anchor, fun_positive, fun_negative

    def Loss(self, pairs, margin=1):
        fun_anchor, fun_positive, fun_negative=self.function(pairs)
        output = self.triplet_loss(fun_anchor, fun_positive, fun_negative)
       # distance_of_pos = F.pairwise_distance(fun_anchor, fun_positive)
       # distance_of_neg=F.pairwise_distance(fun_anchor, fun_negative)
       # tensor_zeros=torch.zeros(distance_of_pos.shape)
       # loss = torch.max(distance_of_pos-distance_of_neg+margin, tensor_zeros)
       # loss=loss.mean()
        return output

    def Train(self, pairs):
        L = self.Loss(pairs)
        self.optimizer.zero_grad()
        L.backward()
        self.optimizer.step()

    def Test(self, pairs):
        anchor, positive, negative = self.function(pairs)
        distance_of_pos = F.pairwise_distance(anchor, positive)
        distance_of_neg = F.pairwise_distance(anchor, negative)
        tensor_zeros = torch.zeros(distance_of_pos.shape, device=self.device)
        margin = 1
        loss = torch.max(distance_of_pos - distance_of_neg + margin, tensor_zeros)
        loss = loss.mean()
        if (torch.cuda.is_available()):
            loss=loss.cpu()
            distance_of_pos=distance_of_pos.cpu()
            distance_of_neg=distance_of_neg.cpu()
        return loss,distance_of_pos, distance_of_neg

    def inference(self, x):
        x = torch.tensor(np.array(x), dtype=torch.float32, device=self.device)
        return self.net(x).cpu().detach().numpy()