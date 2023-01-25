################## used Library  ############################################################
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import os 
import numpy as np
import pandas as pd
import glob
import random
from torch.autograd import Variable
from torch.autograd import Function
from torch import optim
from tdnn import TDNN
import sklearn.metrics

############ number of class and all #####################
n_epoch = 30 # Number of epochs
IP_dim = 80 # number ofinput dimension
#look_back1 = 20 # range

##########################################
############################

def lstm_data(f):

    df = pd.read_csv(f,encoding='utf-16',usecols=list(range(0,IP_dim)))
    dt = df.astype(np.float32)
    X=np.array(dt)
    
    Xdata1 = []
    Xdata2 = []
    Ydata1 = []
      
    mu = X.mean(axis=0)
    std = X.std(axis=0)
    np.place(std, std == 0, 1) 
    X = (X - mu) / std
    Xdata1 = np.array(X)    
    Xdata1 = torch.from_numpy(Xdata1).float()

    return Xdata1


###########################################

################ X_vector #######################



class X_vector(nn.Module):
    def __init__(self, input_dim = 80, num_classes=8):
        super(X_vector, self).__init__()
        self.tdnn1 = TDNN(input_dim=input_dim, output_dim=512, context_size=5, dilation=1,dropout_p=0.5)
        self.tdnn2 = TDNN(input_dim=512, output_dim=512, context_size=3, dilation=1,dropout_p=0.5)
        self.tdnn3 = TDNN(input_dim=512, output_dim=512, context_size=2, dilation=2,dropout_p=0.5)
        self.tdnn4 = TDNN(input_dim=512, output_dim=512, context_size=1, dilation=1,dropout_p=0.5)
        self.tdnn5 = TDNN(input_dim=512, output_dim=512, context_size=1, dilation=3,dropout_p=0.5)
        #### Frame levelPooling
        self.segment6 = nn.Linear(1024, 512)
        self.segment7 = nn.Linear(512, 512)
        self.output = nn.Linear(512, num_classes)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, inputs):
        tdnn1_out = F.relu(self.tdnn1(inputs))
        #print(f'shape of tdnn1 is {tdnn1_out.shape}')
        tdnn2_out = self.tdnn2(tdnn1_out)
        #print(f'shape of tdnn2 is {tdnn2_out.shape}')
        tdnn3_out = self.tdnn3(tdnn2_out)
        #print(f'shape of tdnn3 is {tdnn3_out.shape}')
        tdnn4_out = self.tdnn4(tdnn3_out)
        #print(f'shape of tdnn4 is {tdnn4_out.shape}')
        tdnn5_out = self.tdnn5(tdnn4_out)
        #print(f'shape of tdnn5 is {tdnn5_out.shape}')
        ### Stat Pool
        
        mean = torch.mean(tdnn5_out,1)
        #print(f'shape of mean is {mean.shape}')
        std = torch.var(tdnn5_out,1,)
        #print(f'shape of std is {std.shape}')
        stat_pooling = torch.cat((mean,std),1)
        #print(f'shape of stat_pooling is {stat_pooling.shape}')
        segment6_out = self.segment6(stat_pooling)
        
        segment6_out1 = segment6_out[-1]

        #print(f'shape of segment6 is {segment6_out1.shape}')
        #ht = torch.unsqueeze(ht, 0)
        segment6_out1 = torch.unsqueeze(segment6_out1, 0)
        #print(f'shape of segment6 is {segment6_out1.shape}')
        x_vec = self.segment7(segment6_out1)
        #print(x_vec)
        #print(f'shape of x_vec is {x_vec.shape}')
        predictions = self.output(x_vec)
        #print(predictions)
        #print(f'shape of predictions is {predictions.shape}')
        return predictions

######################## X_vector ####################
##########################################################

def gui(fn):
 
    


    model = X_vector(IP_dim, 8)

    #model.cuda()    
    path = "C:/Users/mkr13/Downloads/gui/gui/5_base1_e18.pth"  ## Load model
    #print(path)
    model.load_state_dict(torch.load(path))

       
    X1 = lstm_data(fn)
    #print(X1.shape)
    XX1 = torch.unsqueeze(X1, 1)
    #print(XX1.shape)
    X1 = np.swapaxes(XX1,0,1)
    #print(X1.shape)
    #x1 = Variable(X1, requires_grad=True).cuda()
    x1 = Variable(X1, requires_grad=True)
    #print(x1.shape)

        

    o1= model.forward(x1)
    #print(o1)
            
            
    Pred = np.argmax(o1.detach().cpu().numpy(),axis=1)
    return Pred
            

    
#    Pred = np.append(Pred,P)
'''     
        CM2=sklearn.metrics.confusion_matrix(Tru, Pred)
        print(CM2)
        txtfl.write(path)
        txtfl.write('\n')
        txtfl.write(str(CM2))
        txtfl.write('\n')
        acc = sklearn.metrics.accuracy_score(Tru,Pred)
        print(acc)
        txtfl.write(str(acc))
        txtfl.write('\n')
        A.append(acc)

    print(A)
    gg=max(A)
    print(gg)
    print(A.index(gg))
    txtfl.write('\n')
    txtfl.write(str(gg))
    txtfl.close()

'''
   
'''
fn = "/home/berlin/Downloads/IITMad_Project_website/sample_BNF/my/3.csv"
val = gui(fn)
print(val)

'''
