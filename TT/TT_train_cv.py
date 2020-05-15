import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

##########################################################
#To speed up training we'll only work on a subset of the data

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from sklearn.metrics import accuracy_score

from numpy import loadtxt

# Load data from https://www.openml.org/d/554
x_test = loadtxt('problem/X_test_regression.csv',delimiter=',')
y_test = loadtxt('problem/y_test_regression.csv',delimiter=',')
x_valid = loadtxt('problem/X_val_regression.csv',delimiter=',')
y_val = loadtxt('problem/y_val_regression.csv',delimiter=',')
x_train = loadtxt('problem/X_train_regression.csv',delimiter=',')#[:128,:]
y_train = loadtxt('problem/y_train_regression.csv',delimiter=',')

train_size = x_train.shape[0]
val_size = x_valid.shape[0]
test_size = x_test.shape[0]

##########################################################
# define network
class Net(nn.Module):

    def __init__(self, num_features, poly_order, rank, num_output, correction_factor, correction_factor_mode2, correction_factor_result):
        super(Net, self).__init__()  
        
        self.num_features = num_features
        self.poly_order = poly_order
        self.num_output = num_output
        self.rank = rank
        self.correction_factor = correction_factor.to(device)
        self.correction_factor_mode2 = correction_factor_mode2.to(device)
        self.correction_factor_result = correction_factor_result.to(device)

        Di = self.rank
        Dn = self.rank
        # Elements are drawn from a uniform distribution [-1/sqrt(D),1/sqrt(D)]
        bound_i = 1/np.sqrt(Di)
        bound_n = 1/np.sqrt(Dn)
        # bounds on the uniform distribution
        lb = 0.5*bound_i
        ub = 1.0*bound_i

        # input layer
        self.tt_cores = []
        #k = np.sqrt(    (1/ (self.poly_order)) * (self.rank)**(  -(self.num_features-1)/self.num_features )  )
        k = 1
        for i in range(num_features):
            if i==0: 
                tn_size = (1,poly_order,self.rank)
            elif i==num_features-1:
                tn_size = (self.rank,poly_order,num_output)
            else:
                tn_size = (self.rank,poly_order,self.rank)

            g_i = Parameter(init.normal_(torch.empty(tn_size, requires_grad=True), mean=0, std=k) * 1/(np.sqrt(self.poly_order)) )#* self.correction_factor[i])
            #g_i = init.normal_(torch.empty(tn_size, requires_grad=True), mean=0, std=1) #* 1/(np.sqrt(self.poly_order)) )
            #g_i = Parameter(F.normalize(g_i.view(g_i.size(0), -1), dim=1, p=2).view(g_i.size()))
            
            #g_i = Parameter(init.uniform_(torch.empty(tn_size, requires_grad=True),a=1,b=1))

            self.tt_cores.append(g_i)
            
        self.tt_cores = nn.ParameterList(self.tt_cores)

    def get_n_params(self):
        pp=0
        for p in list(self.parameters()):
            nn=1
            for s in list(p.size()):
                nn = nn*s
            pp += nn
        return pp


    def get_correction_factor(self, vec_input, batch_size):
                
        vec = vec_input[:,:self.poly_order].reshape(batch_size,-1)
        
        # First do: G_i x_2 v_i
        mode2 = []
        for i in range(self.num_features):
            vec = vec_input[:,i*self.poly_order:(i+1)*self.poly_order].reshape(batch_size,-1)
            #print(' First ', i, torch.mean(vec, 0), torch.std(vec, 0)**2)
            #print(' Secon ', i, torch.mean(self.tt_cores[i]), torch.std(self.tt_cores[i])**2)
            Gi_vi = torch.einsum('abc,db -> dac', self.tt_cores[i], vec) * 1/np.sqrt(self.poly_order)
            #self.correction_factor[i] *= 1/torch.std(Gi_vi)  
            self.correction_factor_mode2[i] = 1/torch.std(Gi_vi)  
            mode2.append(Gi_vi * 1/torch.std(Gi_vi))
            
        mode2[0] = mode2[0].reshape(batch_size,self.rank)
        mode2[-1] = mode2[-1].reshape(batch_size,self.rank,self.num_output)
        
        # Join all the results (based on equation 11 in the paper)
        result = mode2[0]
        for i in range(self.num_features-1):
            #print(i, ' V ', torch.mean(mode2[i+1]), torch.std(mode2[i+1])**2)
            #print(i, ' result ', torch.mean(result), torch.std(result)**2)
            #print(i, result)
            result_by_mode2 = torch.einsum('ab,abd -> ad', result, mode2[i+1]) * 1/np.sqrt(self.rank)
            #self.correction_factor[i] *= 1/torch.std(result_by_mode2)
            self.correction_factor_result[i] = 1/torch.std(result_by_mode2)
            result = result_by_mode2 * 1/torch.std(result_by_mode2)

        #print(' result ', torch.mean(result), torch.std(result)**2)
        return self.correction_factor, self.correction_factor_mode2, self.correction_factor_result

    def forward(self, vec_input, batch_size, print_expr=False):
                
        vec = vec_input[:,:self.poly_order].reshape(batch_size,-1)
        
        # First do: G_i x_2 v_i
        mode2 = []
        for i in range(self.num_features):
            vec = vec_input[:,i*self.poly_order:(i+1)*self.poly_order].reshape(batch_size,-1)
            #print(' First ', i, torch.mean(vec, 0), torch.std(vec, 0)**2)
            #print(' Secon ', i, torch.mean(self.tt_cores[i]), torch.std(self.tt_cores[i])**2)
            mode2.append(torch.einsum('abc,db -> dac', self.tt_cores[i], vec) * 1/np.sqrt(self.poly_order) * self.correction_factor_mode2[i])
            
        mode2[0] = mode2[0].reshape(batch_size,self.rank)
        mode2[-1] = mode2[-1].reshape(batch_size,self.rank,self.num_output)
        
        # Join all the results (based on equation 11 in the paper)
        result = mode2[0]
        for i in range(self.num_features-1):
            #print(i, ' Shapes ', result.size(), mode2[i+1].size())
            #print(i, ' V ', torch.mean(mode2[i+1]), torch.std(mode2[i+1])**2)
            #print(i, ' result ', torch.mean(result), torch.std(result)**2)
            #print(i, result)
            result = torch.einsum('ab,abd -> ad', result, mode2[i+1]) * 1/np.sqrt(self.rank) * self.correction_factor_result[i]
            
        #print(' result forward ', torch.mean(result), torch.std(result)**2)
        #result = torch.einsum('ab,abc -> ac', result, mode2[-1]) * 1/(self.rank)
        
        '''
        X = n_mode_batch(1, self.A, vec , mantain_dim=False, print_expr=print_expr)
        
        for i in range(1,self.num_features):    
            #print(self.A.shape, vec.shape)
            vec = vec_input[:,i*self.poly_order:(i+1)*self.poly_order].reshape(batch_size,-1)
            X = n_mode_batch(2, X, vec, mantain_dim=False, print_expr=print_expr, batch_in_tensor=True)
        '''
        
        return result

##################################    
# PCA


'''
U,S,V = svd(Y,full_matrices=False)
rho = (S*S) / (S*S).sum()
Z = Y@V
Z = Z#[:,:150]
'''

##########################################################
# Create train, val & test

num_features = x_train.shape[1]

##########################################################
#Hyperparameters
ranks = list(range(1,26))
poly_orders = [2,3,4,5,6]
num_classes = 1
num_features = x_train.shape[1]

##########################################################
save_val_loss, save_loss = [], []

for rank in ranks:
    for poly_order in poly_orders:

        ##########################################################
        
        class VV():

            def __init__(self):
                self.mean_train, self.std_train = None, None
                self.selected = None
            def vandermonde_vec(self,dataset, num_instances, num_features, train=True):
                u = np.zeros((num_instances,num_features*poly_order))

                # Get powers
                for row in range(num_instances):
                    for col in range(num_features):
                        u[row,col*poly_order:(col+1)*poly_order] = np.power([dataset[row,col]]*poly_order, list(range(poly_order)))

                if train:
                    N, M = dataset.shape
                    self.selected = np.std(u,0) != 0
                    self.mean_train = u[:,self.selected].mean(axis=0)
                    # Std data
                    Y = u
                    Y[:,self.selected] = u[:,self.selected] - np.ones((N,1))*self.mean_train
                    self.std_train = np.std(Y[:,self.selected],0)
                    #Y = Y[:,np.std(Y,0) != 0]
                    Y[:,self.selected] = Y[:,self.selected]*(1/self.std_train)
                else:
                    N, M = u.shape
                    # Std data
                    Y = u
                    Y[:,self.selected] = u[:,self.selected] - np.ones((N,1))*self.mean_train
                    Y[:,self.selected] = Y[:,self.selected]*(1/self.std_train)
                #u = u.reshape(num_instances,num_features,poly_order)
                return Y

        vv = VV()

        x_train_van = vv.vandermonde_vec(x_train, x_train.shape[0], num_features)
        x_valid_van = vv.vandermonde_vec(x_valid, x_valid.shape[0], num_features, train=False)
        #x_test_van = vandermonde_vec(x_test, x_test.shape[0], num_features, train=False)

        x_train_van = torch.FloatTensor(x_train_van).to(device)
        x_valid_van = torch.FloatTensor(x_valid_van).to(device)
        #x_test_van = torch.FloatTensor(x_test_van).to(device)
            
        ##########################################################
        net = Net(num_features, poly_order, rank, num_classes, torch.ones(num_features), torch.ones(num_features), torch.ones(num_features))
        
        net.to(device)

        ##########################################################
        correction_factor, correction_factor_mode2, correction_factor_result = net.get_correction_factor(Variable(x_train_van), x_train_van.shape[0])
        
        print(net(Variable(x_train_van[:20]), 20, print_expr=True))


        ##########################################################
        #optimizer = optim.LBFGS(net.parameters(), lr=1, max_iter=1, line_search_fn='strong_wolfe')
        optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay = 0.07)
        criterion = nn.MSELoss()


        ##########################################################
        # we could have done this ourselves,
        # but we should be aware of sklearn and it's tools

        # setting hyperparameters and gettings epoch sizes
        batch_size = 64
        num_epochs = 200
        num_samples_train = x_train_van.shape[0]
        num_batches_train = num_samples_train // batch_size
        num_samples_valid = x_valid.shape[0]
        num_batches_valid = num_samples_valid // batch_size
        
        n_train_samples = len(x_train[:,0])
        n_val_samples = len(x_valid[:,0])
        
        # setting up lists for handling loss/accuracy
        losses, val_losses, val_cur_loss = [], [], []

        get_slice = lambda i, size: range(i * size, (i + 1) * size)

        for epoch in range(num_epochs):
            train_acc, train_loss = [], []
            valid_acc, val_loss = [], []
            test_acc, test_loss = [], []
            cur_loss = 0
           
            # Forward -> Backprob -> Update params
            ## Train
            cur_loss = 0
            net.train()
            for i in range(num_batches_train):
                slce = get_slice(i, batch_size)
                x_batch = Variable(x_train_van[slce])

                # compute gradients given loss
                target_batch = Variable(torch.from_numpy(y_train[slce])).to(device)
                
                '''
                def closure():
                    optimizer.zero_grad()
                    output = net(x_batch, batch_size)
                    print(net.tt_cores[0])
                    batch_loss = criterion(output.double(), target_batch.double())
                    #print(output, target_batch)
                    batch_loss.backward(retain_graph=True)
                    return batch_loss

                optimizer.step(closure)
                '''
                 # compute gradients given loss
                output = net(x_batch, batch_size)
                batch_loss = criterion(output.double(), target_batch.double())
                optimizer.zero_grad()
                batch_loss.backward(retain_graph=True)
                optimizer.step()

                '''
                output = net(x_batch, batch_size)
                batch_loss = criterion(output, target_batch)
                cur_loss += batch_loss 
                losses.append(cur_loss / batch_size)
                '''
            net.eval()
            ### Evaluate training
            train_preds, train_targs = [], []
            for i in range(num_batches_train):
                slce = get_slice(i, batch_size)
                x_batch = Variable(x_train_van[slce])

                output = net(x_batch, batch_size)

                train_targs += list(y_train[slce])
                train_preds += list(output.data.cpu().numpy())
    
            batch_loss = criterion(torch.Tensor(np.array(train_targs)), torch.Tensor(np.array(train_preds))).data.cpu().numpy()
            losses.append(batch_loss)
        
            ### Evaluate validation
            val_preds, val_targs = [], []
            for i in range(num_batches_valid):
                slce = get_slice(i, batch_size)
                x_batch = Variable(x_valid_van[slce])

                output = net(x_batch, batch_size)

                val_preds += list(output.data.cpu().numpy())
                val_targs += list(y_val[slce])
            
            batch_loss = criterion(torch.Tensor(np.array(val_preds)), torch.Tensor(np.array(val_targs))).data.cpu().numpy()
            val_losses.append(batch_loss)
            
            if epoch % 1 == 0:
                print("Epoch %2i : Train Loss %f , Valid acc %f" % (
                        epoch+1, losses[-1], val_losses[-1]))
    print()
    print()
    
    save_loss.append(losses)
    save_val_loss.append(val_losses)
    
    
    np.savetxt('results/train_loss_r{}_n{}.csv'.format(rank,poly_order), np.array(loss), delimiter=',')
    np.savetxt('results/val_loss_r{}_n{}.csv'.format(rank,poly_order), np.array(save_val_loss), delimiter=',')

np.savetxt('results/train_loss.csv', np.array(loss), delimiter=',')
np.savetxt('results/val_loss.csv', np.array(save_val_loss), delimiter=',')
