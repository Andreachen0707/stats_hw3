from builtins import range
import numpy as np


def fc_forward(x, w, b):
    out = x.reshape(x.shape[0], w.shape[0]).dot(w) + b
    cache = (x, w, b)
    return out, cache


def fc_backward(dout, cache):
    x, w, b = cache
    dw = x.reshape(x.shape[0], w.shape[0]).T.dot(dout)
    db = np.sum(dout, axis=0)
    dx = dout.dot(w.T).reshape(x.shape)
    return dx, dw, db


def relu_forward(x):
    out = np.maximum(0, x)
    cache = x
    return out, cache


def relu_backward(dout, cache):
    x = cache
    dx = dout
    dx[cache < 0] = 0
    return dx

def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    pad = conv_param['pad']
    stride = conv_param['stride']
    N,C,H,W = x.shape
    F,C,HH,WW = w.shape
    H_o = 1 + (H + 2 * pad - HH) // stride
    W_o = 1 + (W + 2 * pad - WW) // stride
    window_input=np.zeros((N,F,C,H_o,W_o))
    out = np.zeros((N,F,H_o,W_o))
    
    
    def pad_with(vector, pad_width, iaxis, kwargs):
         pad_value = kwargs.get('padder', 0)
         vector[:pad_width[0]] = pad_value
         vector[-pad_width[1]:] = pad_value
         return vector

    for i in range(0,N):
        for f in range(0,F):
            for m in range(0,H_o):
                hs = m*stride
                for n in range(0,W_o):
                    ws = n*stride
                    for c in range(0,C):
                        x_pad = np.pad(x[i,c,:,:],pad,pad_with)
                        window_input[i,f,c,m,n] += np.sum(x_pad[hs:hs+HH,ws:ws+WW]*w[f,c,:,:])
                    out[i,f,m,n] = np.sum(window_input[i,f,:,m,n])
            out[i,f,:,:]+=b[f]


           
            
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    x,w,b,conv_param = cache
    pad = conv_param['pad']
    stride = conv_param['stride']
    N,C,H,W = x.shape
    F,C,HH,WW = w.shape
    H_o = 1 + (H + 2 * pad - HH) // stride
    W_o = 1 + (W + 2 * pad - WW) // stride
    
    x_pad = np.zeros((N,C,H+2*pad,W+2*pad))
    dx_pad = np.zeros(x_pad.shape)
    dx = np.zeros(x.shape)
    dw = np.zeros(w.shape)
    db = np.zeros(b.shape)
    
    def pad_with(vector, pad_width, iaxis, kwargs):
         pad_value = kwargs.get('padder', 0)
         vector[:pad_width[0]] = pad_value
         vector[-pad_width[1]:] = pad_value
         return vector

    for i in range(0,N):
        for c in range(0,C):
            x_pad[i,c,:,:] = np.pad(x[i,c,:,:],pad,pad_with)
            dx_pad[i,c,:,:] = np.pad(dx[i,c,:,:],pad,pad_with)
            
    for i in range(0,N):
        for f in range(0,F):
            for m in range(0,H_o):
                hs = m*stride
                for n in range(0,W_o):
                    ws = n*stride
                    window_input = x_pad[i,:,hs:hs+HH,ws:ws+WW]
                    dw[f]+=dout[i,f,m,n]*window_input
                    db[f] +=dout[i,f,m,n]
                    dx_pad[i,:,hs:hs+HH,ws:ws+WW]+=w[f,:,:,:]*dout[i,f,m,n]

    dx = dx_pad[:,:,pad:pad+H,pad:pad+W]
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max pooling forward pass                            #
    ###########################################################################
    N,C,H,W = x.shape
    ph = pool_param['pool_height']
    pw = pool_param['pool_width']
    stride = pool_param['stride']

    H_o = 1 + (H  - ph) // stride
    W_o = 1 + (W  - pw) // stride
    out = np.zeros((N,C,H_o,W_o))

    for i in range(N):
        for c in range(C):
            for m in range(H_o):
                hs = m*stride
                for n in range(W_o):
                    ws = n*stride
                    window = x[i,c,hs:hs+ph,ws:ws+pw]
                    out[i,c,m,n] = np.max(window)
                    




    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max pooling backward pass                           #
    ###########################################################################
    x,pool_param = cache
    N,C,H,W = x.shape
    ph = pool_param['pool_height']
    pw = pool_param['pool_width']
    stride = pool_param['stride']

    H_o = 1 + (H  - ph) // stride
    W_o = 1 + (W  - pw) // stride
    dx = np.zeros(x.shape)

    for i in range(N):
        for c in range(C):
            for m in range(H_o):
                hs = m*stride
                for n in range(W_o):
                    ws = n*stride
                    window = x[i,c,hs:hs+ph,ws:ws+pw]
                    max_out = np.max(window)
                    dx[i,c,hs:hs+ph,ws:ws+pw] += (window == max_out)*dout[i,c,m,n]
                    
    
    
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx

def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
