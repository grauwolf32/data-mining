import argparse
import numpy as np
import matplotlib.pyplot as pl

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from numpy import linalg

__author__ = 'vladimirgulin'

def generate_random_data(average, dim, items):
    ''' Generate data from normal distribution

    Arguments:
    average - center of normal distribution
    dim - dimension of data
    items - number of items, that will be generated
    '''

    list_vals = []

    for i in range(dim):
        list_vals.append(average)

    mu_vec = np.array(list_vals)

    cov_matrix = np.identity(dim)

    index = 1

    while index + 1 < cov_matrix.shape[0]:
        jndex = index + 1
        while jndex < cov_matrix.shape[0]:
            gen_val = np.random.uniform(0, 1)
            cov_matrix[index][jndex] = gen_val
            cov_matrix[jndex][index] = gen_val
            jndex = jndex + 1
        index = index + 1

    return np.random.multivariate_normal(mu_vec, cov_matrix, items).T


def simple_projection(total_data):
    ''' Computes simple projections of data (just use first two cordinats)

    Arguments:
    total_data - all observed data
    '''

    sprojection = total_data[0:2, :]

    return sprojection


def compute_pca(total_data):
    ''' This function  computes projection on first two eigen vectors of covarience matrix

    Arguments:
    total_data - all observed data
    '''
    x_mean = np.array(np.mean(total_data,axis=1))
    n,m = total_data.shape
    R = np.array(np.zeros((n,n)))
    
    for i in xrange(0,m):
		R = R + np.outer(np.array(total_data[:,i]-x_mean),np.array(total_data[:,i]-x_mean))
	
    R = R*(1.0/float(m))
    lbd,vectors = linalg.eig(R)

    lbd_sum = sum(lbd)
    summ = 0.0
    best_dimm = 0

    for i in xrange(len(lbd)):
        summ += lbd[i]
        best_dimm += 1
        if summ/lbd_sum > 0.95:
	       break
    print 'dimm:'
    print n
    print 'Best dimm: '
    print best_dimm
    print 'pr: '
    print summ/lbd_sum
  
    B = np.dot(x_mean.T,vectors)
    P = np.dot(np.array(total_data).T,vectors)
    P = np.array(P)
    return P[:,0:best_dimm].T



def main():
    np.random.seed(234234782384) # random seed for consistency

    args = parse_args()

    class_size = args.items

    generated_data1 = generate_random_data(0, 3, class_size)
    generated_data2 = generate_random_data(2, 3, class_size)
    generated_data3 = generate_random_data(-2, 3, class_size)

    fig = pl.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    pl.rcParams['legend.fontsize'] = 10
    ax.plot(generated_data1[0,:], generated_data1[1,:],\
    generated_data1[2,:], 'o', markersize=8, color='blue', alpha=0.5, label='class1')
    ax.plot(generated_data2[0,:], generated_data2[1,:],\
    generated_data2[2,:], 'o', markersize=8, color='red', alpha=0.5, label='class2')
    ax.plot(generated_data3[0,:], generated_data3[1,:],\
    generated_data3[2,:], 'o', markersize=8, color='green', alpha=0.5, label='class3')

    pl.title('Original 3d data')
    ax.legend(loc='upper right')
    pl.show()
    
    total_data = np.concatenate(([generated_data1, generated_data2, generated_data3]), axis=1)
    
    #Simple projection
    pc = simple_projection(total_data)

    pl.plot(pc[0, 0:class_size], pc[1, 0:class_size],\
         'o', markersize=7, color='blue', alpha=0.5, label='class1')

    pl.plot(pc[0, class_size: 2 * class_size], pc[1, class_size: 2 * class_size],\
         'o', markersize=7, color='red', alpha=0.5, label='class2')

    pl.plot(pc[0, 2 * class_size: 3 * class_size], pc[1, 2 * class_size: 3 * class_size],\
         'o', markersize=7, color='green', alpha=0.5, label='class3')

    pl.title('Projected data')

    pl.show()
    
	#PCA
    pc = compute_pca(total_data)

    pl.plot(pc[0, 0:class_size], pc[1, 0:class_size],\
         'o', markersize=7, color='blue', alpha=0.5, label='class1')

    pl.plot(pc[0, class_size: 2 * class_size], pc[1, class_size: 2 * class_size],\
         'o', markersize=7, color='red', alpha=0.5, label='class2')

    pl.plot(pc[0, 2 * class_size: 3 * class_size], pc[1, 2 * class_size: 3 * class_size],\
         'o', markersize=7, color='green', alpha=0.5, label='class3')

    pl.title('PCA data')

    pl.show()
    #Multidimensional data generation
    generated_data1 = generate_random_data(0, 50, class_size)
    generated_data2 = generate_random_data(1, 50, class_size)
    generated_data3 = generate_random_data(-1, 50, class_size)

    total_data = np.concatenate(([generated_data1, generated_data2, generated_data3]), axis=1)
    print 'Multidimensional data was generated...'
    pc = compute_pca(total_data)    
    
   


def parse_args():
    parser = argparse.ArgumentParser(description='Principal component analysis')
    parser.add_argument("-i", "--items", action="store", type=int, help="Number of items for each class", default=25)
    return parser.parse_args()


if __name__ == "__main__":
    main()
