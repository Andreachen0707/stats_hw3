from builtins import object
import numpy as np

from stats232a.layers import *
from stats232a.fast_layers import *
from stats232a.layer_utils import *

class OneBlockResnet(object):
    """
    A convolutional network with a residual block:
    conv - relu - 2x2 max pool - residual block - relu - fc - relu - fc - softmax
    The residual block has the following structure:
           ______________________      
          |                      |
    input - conv - relu - conv - + - output
    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=[64, 64, 64], filter_size=[7, 3, 3], 
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.
        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in each convolutional layer.
        - filter_size: Size of filters to use in each convolutional layer.
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final fc layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the one block residual network.  #
        # Weights should be initialized from a Gaussian with standard deviation    #
        # equal to weight_scale; biases should be initialized to zero.             #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2', and keys 'W3' and 'b3' for the two     #
        # convolutional layers in residual block; use keys 'W4' and 'b4' for the   #
        # hidden fc layer; use keys 'W5' and 'b5' for the output fc layer.         #
        ############################################################################

        self.params['W1'] = np.random.normal(scale = weight_scale,size = (num_filters[0],input_dim[0],filter_size[0],filter_size[0]))
        self.params['W2'] = np.random.normal(scale = weight_scale,size = (num_filters[1],num_filters[0],filter_size[1],filter_size[1]))
        self.params['W3'] = np.random.normal(scale = weight_scale,size = (num_filters[2],num_filters[1],filter_size[2],filter_size[2]))
        self.params['W4'] = np.random.normal(scale = weight_scale,size = (num_filters[2]*input_dim[1]*input_dim[2]//4,hidden_dim))
        self.params['W5'] = np.random.normal(scale = weight_scale,size = (hidden_dim,num_classes))
        self.params['b1'] = np.zeros(num_filters[0])
        self.params['b2'] = np.zeros(num_filters[1])
        self.params['b3'] = np.zeros(num_filters[2])
        self.params['b4'] = np.zeros(hidden_dim)
        self.params['b5'] = np.zeros(num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.
        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        W4, b4 = self.params['W4'], self.params['b4']
        W5, b5 = self.params['W5'], self.params['b5']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param1 = {'stride': 1, 'pad': (filter_size - 1) // 2}
        
        filter_size = W2.shape[2]
        conv_param2 = {'stride': 1, 'pad': (filter_size - 1) // 2}
        
        filter_size = W3.shape[2]
        conv_param3 = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the one block residual net,         #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        score_1,cache_1 = conv_relu_pool_forward(X, W1, b1, conv_param1,pool_param)
        score_2,cache_2 = conv_relu_forward(score_1, W2, b2, conv_param2)
        score_3,cache_3 = conv_forward_fast(score_2, W3, b3, conv_param3)
        score_3_rl, cache_3_rl = relu_forward(score_3+score_1)
        score_4,cache_4 = fc_relu_forward(score_3_rl,W4,b4)
        scores,cache_5 = fc_forward(score_4,W5,b5)
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the one block residual net,        #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        loss,dout = softmax_loss(scores,y)
        dx_1, grads['W5'],grads['b5'] = fc_backward(dout,cache_5)
        dx_2, grads['W4'],grads['b4'] = fc_relu_backward(dx_1,cache_4)
        dx_2_rl = relu_backward(dx_2,cache_3_rl)
        dx_3, grads['W3'],grads['b3'] = conv_backward_fast(dx_2_rl,cache_3)
        dx_4, grads['W2'],grads['b2'] = conv_relu_backward(dx_3,cache_2)
        dx_5, grads['W1'],grads['b1'] = conv_relu_pool_backward(dx_4+dx_2,cache_1)

        loss = loss + 0.5*self.reg*np.sum(W1*W1)+ 0.5*self.reg*np.sum(W2*W2)+0.5*self.reg*np.sum(W3*W3)+0.5*self.reg*np.sum(W4*W4)+0.5*self.reg*np.sum(W5*W5)
        grads['W1'] += self.reg*W1
        grads['W2'] += self.reg*W2
        grads['W3'] += self.reg*W3
        grads['W4'] += self.reg*W4
        grads['W5'] += self.reg*W5
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
