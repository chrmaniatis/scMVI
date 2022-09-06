#File containing functions for creating scatter plots.
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
from matplotlib import cm
from matplotlib.colors import Normalize 
import scipy.stats as stats
from scipy.stats import gaussian_kde
import numpy as np
import tensorflow as tf

def density_scatter_plot(x,y,err, **kwargs):
    """
    :param x: data positions on the x axis
    :param y: data positions on the y axis
    :return: matplotlib.collections.PathCollection object
    """
    # Kernel Density Estimate (KDE)
    values = np.vstack((x, y))
    kernel = gaussian_kde(values)
    kde = kernel.evaluate(values)

    # create array with colors for each data point
    norm = Normalize(vmin=kde.min(), vmax=kde.max())
    colors = cm.ScalarMappable(norm=norm, cmap='viridis').to_rgba(kde)

    # override original color argument
    kwargs['color'] = colors

    mm=tf.math.maximum(tf.reduce_max(x),tf.reduce_max(y))
    xx=tf.range(0, mm+1, 1)

    plt.scatter(x, y, **kwargs)
    plt.colorbar(orientation='vertical')
    plt.plot(xx,xx,color='red')
    

    plt.errorbar(x,y, yerr=err,ecolor=colors , marker = '', ls = '', zorder = 0)
    return plt

def met_scatter_plot(x,y,cvv,err, **kwargs):
              mm=tf.math.maximum(tf.reduce_max(x),tf.reduce_max(y))
              xx=tf.range(0, mm+1, 1)
        
              sct = plt.scatter(x, y,c=cvv ,**kwargs)
              plt.plot(xx,xx,color='red')
        
              colors = cvv
              colors -= np.min(colors)
              colors *=  (1./np.max(colors))
        
              _, _ , errorlinecollection = plt.errorbar(x,y, yerr=err, marker = '', ls = '', zorder = 0)
              error_color = sct.to_rgba(colors)
        
              errorlinecollection[0].set_color(error_color)
        
              return plt