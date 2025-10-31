import pickle as pkl
from scipy.stats import uniform
import numpy as np
import os
import g_tr as gt


N = 10000
dataname = '10000pts'



domain_data_x = uniform.rvs(size=N)
domain_data_y = uniform.rvs(size=N)

domain_data = np.array([domain_data_x,domain_data_y]).T
print(domain_data.shape)


Nb = 3000

def generate_random_bdry(Nb):
    '''
    Generate random boundary points.
    '''
    bdry_col = uniform.rvs(size=Nb*2).reshape([Nb,2])
    for i in range(Nb):
        randind = np.random.randint(0,2)
        if bdry_col[i,randind] <= 0.5:
            bdry_col[i,randind] = 0.0
        else:
            bdry_col[i,randind] = 1.0

    return bdry_col

bdry_col = generate_random_bdry(Nb)




if not os.path.exists('dataset/'):
    os.makedirs('dataset/')
with open('dataset/'+dataname,'wb') as pfile:
    pkl.dump(domain_data,pfile)
    pkl.dump(bdry_col,pfile)


ygt,fgt = gt.data_gen_interior(domain_data)
bdry_dat = gt.data_gen_bdry(bdry_col)



with open("dataset/gt_on_{}".format(dataname),'wb') as pfile:
    pkl.dump(ygt,pfile)
    pkl.dump(fgt,pfile)
    pkl.dump(bdry_dat,pfile)

   