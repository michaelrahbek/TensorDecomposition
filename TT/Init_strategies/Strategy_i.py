# define network
class Net(nn.Module):

    def __init__(self, num_features, poly_order, num_output):
        super(Net, self).__init__()  
        
        self.num_features = num_features
        self.poly_order = poly_order
        self.num_output = num_output
        self.rank = 3

        # input layer
        self.tt_cores = []
        k = np.sqrt(    (1/ (self.poly_order)) * (self.rank)**(  -(self.num_features-1)/self.num_features )  )
        for i in range(num_features):
            if i==0: 
                tn_size = (1,poly_order,self.rank)
            elif i==num_features-1:
                tn_size = (self.rank,poly_order,num_output)
            else:
                tn_size = (self.rank,poly_order,self.rank)

            g_i = Parameter(init.normal_(torch.empty(tn_size, requires_grad=True), mean=0, std=k))
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

    def forward(self, vec_input, batch_size, print_expr=False):
                
        vec = vec_input[:,:self.poly_order].reshape(batch_size,-1)
        
        # First do: G_i x_2 v_i
        mode2 = []
        for i in range(self.num_features):
            vec = vec_input[:,i*self.poly_order:(i+1)*self.poly_order].reshape(batch_size,-1)
            mode2.append(torch.einsum('abc,db -> dac', self.tt_cores[i], vec))
            
        mode2[0] = mode2[0].reshape(batch_size,self.rank)
        mode2[-1] = mode2[-1].reshape(batch_size,self.rank,self.num_output)
        
        # Join all the results (based on equation 11 in the paper)
        result = mode2[0]
        for i in range(self.num_features-1):
            result = torch.einsum('ab,abd -> ad', result, mode2[i+1])

        
        return result