# model.py

from collections import OrderedDict
import torch.nn as nn
from torch import rand, Tensor
import torch.optim as optim
import h5py
import torch
import numpy as np

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        
        self.base = self.get_base()
        if args['trained']['use']:
            self.base.load_state_dict(
                self.get_weight_dict(),
                strict=False
            )
            for param in self.base.parameters():
                param.requires_grad = False

        self.conv_fc_size = self.get_conv_fc_size()
        self.conv_head = self.get_conv_head()

        self.surv_fc_size = self.get_surv_fc_size()
        self.surv_head = self.get_surv_head()
        
    def get_base(self):
        base = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))),
            ('relu1', nn.ReLU()),
            ('pool1', nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))),
            ('conv2', nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))),
            ('relu2', nn.ReLU()),
            ('pool2', nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))),
            ('conv3a', nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))),
            ('relu3a', nn.ReLU()),
            ('conv3b', nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))),
            ('relu3b', nn.ReLU()),
            ('pool3', nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))),   
            ('conv4a', nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))),
            ('relu4a', nn.ReLU()),
            ('conv4b', nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))),
            ('relu4b', nn.ReLU()),
            ('pool4 ', nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))),
            ('conv5a', nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))),
            ('relu5a', nn.ReLU()),
            ('conv5b', nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))),
            ('relu5b', nn.ReLU()),
            ('pool5', nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))),
            ('flatten', nn.Flatten(start_dim=1, end_dim=-1))
        ]))
        
        return base
    
    def get_conv_head(self):
        head_dict = OrderedDict()
        size = self.conv_fc_size
        args = self.args['size']['conv_head']
        ii = 0
        for units, drop in zip(args['dense_units'], args['dense_drop']):
            head_dict[f'fc{ii}'] = nn.Linear(size, units)
            size = units
            head_dict[f'relu{ii}'] = nn.ReLU()
            if drop:
                head_dict[f'drop{ii}'] = nn.Dropout(p=drop)
            ii += 1

        self.fc_count = ii

        return nn.Sequential(head_dict)
            
    def get_surv_head(self):
        head_dict = OrderedDict()
        size = self.surv_fc_size
        args = self.args['size']['surv_head']
        ii = self.fc_count
        for units, drop in zip(args['dense_units'], args['dense_drop']):
            head_dict[f'fc{ii}'] = nn.Linear(size, units)
            size = units
            head_dict[f'relu{ii}'] = nn.ReLU()
            if drop:
                head_dict[f'drop{ii}'] = nn.Dropout(p=drop)
            ii += 1
            
        if args['out_act'] == 'non_prop':
            logits = nn.Linear(size, self.args['size']['output_size'])
            logits.weight.data.fill_(0.0)
            logits.bias.data.fill_(0.0)
            head_dict['logits'] = logits
            head_dict['probs'] = nn.Sigmoid()
        elif args['out_act'] == 'prop':
            logits = nn.Linear(size, 1, bias=False)
            logits.weight.data.fill_(0.0)
            head_dict['logits'] = logits
            head_dict['probs'] = self.PropHazards(1, self.args['size']['output_size']) 
        else:
            raise ValueError('out_act not listed')

        return nn.Sequential(head_dict)
            
            
    def forward(self, ct, pt, clin):
        for layer in self.base:
            ct = layer(ct)
            pt = layer(pt)

        x = torch.cat((ct, pt), dim=-1)

        for layer in self.conv_head:
            x = layer(x)

        x = torch.cat((x, clin), dim=-1)

        for layer in self.surv_head:
            x = layer(x)
            
        return x
    
    def get_conv_fc_size(self):
        conv_size = self.base(
            rand((1, *self.args['size']['input_size']))
        ).shape[-1]
        return 2*conv_size
    
    def get_surv_fc_size(self):
        cat_size = self.args['size']['cat_size']
        conv_size = self.conv_head(rand(1, self.conv_fc_size)).shape[-1]
        return cat_size + conv_size
    
    def get_weight_dict(self):
        weight_dict = OrderedDict()
        with h5py.File(self.args['trained']['path'], 'r') as ff:
            layer = list(ff.keys())
            for ll in layer:
                if ll.startswith('conv') or ll.startswith('fc'):
                    try:
                        weight = ff[ll][ll]['kernel:0'][:]
                        bias = ff[ll][ll]['bias:0'][:]
                    except:
                        weight = ff[ll]['kernel'][:]
                        bias = ff[ll]['bias'][:]
                    weight_dict[ll+'.weight'] = Tensor(weight.T)
                    weight_dict[ll+'.bias'] = Tensor(bias.T)
        return weight_dict
    
    def get_optimizer(self):
        args = self.args['optimizer']
        if args['name'].lower() == 'adam':
            optimizer = optim.Adam
        elif args['name'].lower() == 'sgd':
            optimizer = optim.SGD
        else:
            raise ValueError('optimizer not listed')
        
        names = optimizer.__init__.__code__.co_varnames[1:-1]
        values = list(optimizer.__init__.__defaults__)
        variables = dict(zip(names[1:], values))
        variables['params'] = self.parameters()
    
        for ag in args.keys():
            if ag == 'name':
                continue
            if ag in names:
                val = args[ag]
                if isinstance(val, str) and 'e-' in str(val):
                    val = float(val)
                variables[ag] = val 
            else:
                raise ValueError(f'variable {ag} not in callback')
   
        return optimizer(**variables)

    def get_loss(self):
        if self.args['loss']['type'].lower() == 'nnet_survival':
            loss = self.surv_likelihood(self.args['size']['output_size'])
        else:
            raise ValueError('loss not listed')

        return loss

    # nnet_survival part
    # following the model in: https://github.com/MGensheimer/nnet-survival
    
    @staticmethod
    def surv_likelihood(n_intervals):
        """
        Arguments
            n_intervals: the number of survival time intervals
        Returns
            Custom loss function
        """
        def loss(y_pred, y_true):
            """
            Arguments
                y_true: Tensor.
                  First half of the values is 1 if individual survived that interval, 0 if not.
                  Second half of the values is for individuals who failed, and is 1 for time interval during which failure occured, 0 for other intervals.
                  See make_surv_array function.
                y_pred: Tensor, predicted survival probability (1-hazard probability) for each time interval.
            Returns
                Vector of losses for this minibatch.
            """
            cens_uncens = 1. + y_true[:,0:n_intervals] * (y_pred-1.) #component for all individuals
            uncens = 1. - y_true[:,n_intervals:2*n_intervals] * y_pred #component for only uncensored individuals
            epsilon = 1e-7
            loglik = torch.sum(-torch.log(torch.clip(torch.cat((cens_uncens,uncens), dim=-1), epsilon ,None)), axis=-1) #return -log likelihood
            return torch.mean(loglik)
                
        return loss

    class PropHazards(nn.Module):

        def __init__(self, size_in, size_out):
            super().__init__()
            self.size_in, self.size_out = size_in, size_out
            weights = torch.zeros(size_out, size_in)
            self.weights = nn.Parameter(weights)

        def forward(self, x):
            return torch.pow(torch.sigmoid(self.weights.t()), torch.exp(x))

def make_surv_array(t,f,breaks):
    """Transforms censored survival data into vector format
      Arguments
          t: Array of failure/censoring times.
          f: Censoring indicator. 1 if failed, 0 if censored.
          breaks: Locations of breaks between time intervals for discrete-time survival model (always includes 0)
      Returns
          Two-dimensional array of survival data, dimensions are number of individuals X number of time intervals*2
    """
    n_samples=t.shape[0]
    n_intervals=len(breaks)-1
    timegap = breaks[1:] - breaks[:-1]
    breaks_midpoint = breaks[:-1] + 0.5*timegap
    y_train = np.zeros((n_samples,n_intervals*2))
    for i in range(n_samples):
        if f[i]: #if failed (not censored)
            y_train[i,0:n_intervals] = 1.0*(t[i]>=breaks[1:]) #give credit for surviving each time interval where failure time >= upper limit
            if t[i]<breaks[-1]: #if failure time is greater than end of last time interval, no time interval will have failure marked
                y_train[i,n_intervals+np.where(t[i]<breaks[1:])[0][0]]=1 #mark failure at first bin where survival time < upper break-point
        else: #if censored
            y_train[i,0:n_intervals] = 1.0*(t[i]>=breaks_midpoint) #if censored and lived more than half-way through interval, give credit for surviving the interval.
    return y_train
