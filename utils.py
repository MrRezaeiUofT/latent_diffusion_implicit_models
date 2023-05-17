

import numpy as np
from scipy.ndimage import filters
from scipy.signal.windows import gaussian
import torch
from scipy import stats
from sklearn import metrics
def gaussian_kernel_smoother(y, sigma, window):
    b = gaussian(window, sigma)
    y_smooth = np.zeros(y.shape)
    neurons = y.shape[1]
    for neuron in range(neurons):
        y_smooth[:, neuron] = np.convolve(y[:, neuron], b/b.sum(),'same')
    return y_smooth



def get_diagonal(matrix):
    output=torch.zeros((matrix.shape[0],matrix.shape[1]))
    for ii in range( matrix.shape[0]):
        output[ii,:] = torch.diagonal(torch.squeeze(matrix[ii]))
    return output

def calDesignMatrix_V2(X,h):
    '''

    :param X: [samples*Feature]
    :param h: hist
    :return: [samples*hist*Feature]

    '''
    PadX = np.zeros([h , X.shape[1]])
    PadX =np.concatenate([PadX,X],axis=0)
    XDsgn=np.zeros([X.shape[0], h, X.shape[1]])
    # print(PadX.shapepe)
    for i in range(0,XDsgn.shape[0]):
         #print(i)
         XDsgn[i, : , :]= (PadX[i:h+i,:])
    return XDsgn

def calSmoothNeuralActivity(data,gausWindowLength,gausWindowSigma):
    x=np.linspace(-1*gausWindowSigma,1*gausWindowSigma,gausWindowLength)
    gausWindow=1/(2*np.pi*gausWindowSigma)*np.exp(-0.5*(x**2/gausWindowSigma**2))
    gausWindow=gausWindow/np.max(gausWindow)
    #plt.plot(x,gausWindow)
    #plt.show()
    dataSmooth=np.zeros(data.shape)
    for i in range(data.shape[1]):
        dataSmooth[:,i]=np.convolve(data[:,i],gausWindow,'same')
        #dataSmooth[np.where(dataSmooth[:,i] <0), i]=0
    #plt.subplot(2,1,1)
    #plt.plot(data[:5000,1])
    #plt.subplot(2, 1, 2)
    #plt.plot(dataSmooth[:5000, 1])
    #plt.show()
    return dataSmooth
def calInformetiveChan(data,minNumSpiks):
    return np.where(np.sum(data,axis=0)>minNumSpiks)

def calDesignMatrix_V2(X,h):
    '''

    :param X: [samples*Feature]
    :param h: hist
    :return: [samples*hist*Feature]

    '''
    PadX = np.zeros([h , X.shape[1]])
    PadX =np.concatenate([PadX,X],axis=0)
    XDsgn=np.zeros([X.shape[0], h* X.shape[1]])
    # print(PadX.shapepe)
    for i in range(0,XDsgn.shape[0]):
         #print(i)
         XDsgn[i,  :]= (PadX[i:h+i,:]).reshape([-1,])
    return XDsgn

def calSmoothNeuralActivity(data,gausWindowLength,gausWindowSigma):
    x=np.linspace(-1*gausWindowSigma,1*gausWindowSigma,gausWindowLength)
    gausWindow=1/(2*np.pi*gausWindowSigma)*np.exp(-0.5*(x**2/gausWindowSigma**2))
    gausWindow=gausWindow/np.max(gausWindow)
    #plt.plot(x,gausWindow)
    #plt.show()
    dataSmooth=np.zeros(data.shape)
    for i in range(data.shape[1]):
        dataSmooth[:,i]=np.convolve(data[:,i],gausWindow,'same')
        #dataSmooth[np.where(dataSmooth[:,i] <0), i]=0
    #plt.subplot(2,1,1)
    #plt.plot(data[:5000,1])
    #plt.subplot(2, 1, 2)
    #plt.plot(dataSmooth[:5000, 1])
    #plt.show()
    return dataSmooth
def calInformetiveChan(data,minNumSpiks):
    return np.where(np.sum(data,axis=0)>minNumSpiks)

def get_normalized(x, config):
    if config['supervised']:
        return (x-x.mean())/x.std()
    else:
        return  (x-x.mean())/x.std()

def get_cdf(data):
    count, bins_count = np.histogram(data, bins=10)

    # finding the PDF of the histogram using count values
    pdf = count / sum(count)

    # using numpy np.cumsum to calculate the CDF
    # We can also find using the PDF values by looping and adding
    cdf = np.cumsum(pdf)
    return cdf

def get_metrics(z, z_hat):
    cc=[]
    mse=[]
    mae=[]
    for ii in range(z.shape[1]):
        cc.append(stats.pearsonr(z[:,ii],z_hat[:,ii])[0])
        mse.append(metrics.mean_squared_error(z[:, ii], z_hat[:, ii]))
        mae.append(metrics.mean_absolute_error(z[:, ii], z_hat[:, ii]))

    return np.mean(cc), np.mean(mse),np.mean(mae)