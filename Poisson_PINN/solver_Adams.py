from time import time
from tracemalloc import start
import numpy as np
import torch
import torch.optim as opt
import matplotlib.pyplot as plt
import torch.utils.data.dataset as Dataset
import torch.utils.data.dataloader as Dataloader
from torch.autograd import Variable
import pickle as pkl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import model,pde,data,tools,g_tr,validation
from time import time

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


torch.set_default_dtype(torch.float32)

y = model.NN()
y.apply(model.init_weights)

dataname = '10000pts'
name = 'results/'

bw = 60.0

if not os.path.exists(name):
    os.makedirs(name)

if not os.path.exists(name+"y_plot/"):
    os.makedirs(name+"y_plot/")


params = list(y.parameters())



with open("dataset/"+dataname,'rb') as pfile:
    int_col = pkl.load(pfile)
    bdry_col = pkl.load(pfile)

print(int_col.shape,bdry_col.shape)

intx1,intx2 = np.split(int_col,2,axis=1)
bdx1,bdx2 = np.split(bdry_col,2,axis=1)

tintx1,tintx2,tbdx1,tbdx2 = tools.from_numpy_to_tensor([intx1,intx2,bdx1,bdx2],[True,True,False,False,],dtype=torch.float32)


with open("dataset/gt_on_{}".format(dataname),'rb') as pfile:
    y_gt = pkl.load(pfile)
    f_np = pkl.load(pfile)
    bdry_np = pkl.load(pfile)

f,bdrydat,ygt = tools.from_numpy_to_tensor([f_np,bdry_np,y_gt],[False,False,False],dtype=torch.float32)


optimizer = opt.Adam(params,lr=1e-4)

mse_loss = torch.nn.MSELoss()

scheduler = opt.lr_scheduler.ReduceLROnPlateau(optimizer,patience=500)
loader = torch.utils.data.DataLoader([intx1,intx2],batch_size = 500,shuffle = True)


def closure():
    tot_loss = 0
    for i,subquad in enumerate(loader):
       optimizer.zero_grad()
       ttintx1 = Variable(subquad[0].float(),requires_grad = True)
       ttintx2 = Variable(subquad[1].float(),requires_grad = True)

       loss,pres,bres = pde.pdeloss(y,ttintx1,ttintx2,f,tbdx1,tbdx2,bdrydat,bw)
       loss.backward()
       optimizer.step()
       tot_loss = tot_loss + loss

    nploss = tot_loss.detach().numpy()
    scheduler.step(nploss)
    return nploss   



  

losslist = list()

start = time()
for epoch in range(200):
    loss = closure()
    losslist.append(loss)
    if epoch %100==0:
        end = time()
        wall_clock = end-start
        print("epoch: {}, loss:{}, time:{}".format(epoch,loss,wall_clock))
        validation.plot_2D(y,name+"y_plot/"+'epoch{}'.format(epoch))
        
end = time()
with open("results/losshist.pkl",'wb') as pfile:
    pkl.dump(losslist,pfile)








x_pts = np.linspace(0,1,200)
y_pts = np.linspace(0,1,200)

ms_x, ms_y = np.meshgrid(x_pts,y_pts)

x_pts = np.ravel(ms_x).reshape(-1,1)
t_pts = np.ravel(ms_y).reshape(-1,1)

collocations = np.concatenate([x_pts,t_pts], axis=1)

u_gt1,f = g_tr.data_gen_interior(collocations)
#u_gt1 = [np.sin(np.pi*x_col)*np.sin(np.pi*y_col) for x_col,y_col in zip(x_pts,t_pts)]
#u_gt1 = [np.exp(x_col+y_col) for x_col,y_col in zip(x_pts,t_pts)]

#u_gt = np.array(u_gt1)

ms_ugt = u_gt1.reshape(ms_x.shape)

pt_x = Variable(torch.from_numpy(x_pts).float(),requires_grad=False)
pt_t = Variable(torch.from_numpy(t_pts).float(),requires_grad=False)

pt_y = y(pt_x,pt_t)
y = pt_y.data.cpu().numpy()
ms_ysol = y.reshape(ms_x.shape)




import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plotting
import numpy as np

# Example: assuming ms_x, ms_y, ms_ysol are 2D arrays (same shape)
# If ms_x, ms_y are 1D, convert them using np.meshgrid
# ms_x, ms_y = np.meshgrid(ms_x, ms_y)

fig_1 = plt.figure(1, figsize=(6, 5))
ax = fig_1.add_subplot(111, projection='3d')

surf = ax.plot_surface(ms_x, ms_y, ms_ysol, cmap='jet', edgecolor='none')

# Colorbar
h = fig_1.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
h.ax.tick_params(labelsize=14)

# Labels and formatting
ax.set_xlabel('x', fontsize=16)
ax.set_ylabel('y', fontsize=16)
ax.set_zlabel('u(x,y)', fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=12)

plt.tight_layout()
plt.savefig('NNsolution_surface.png', bbox_inches='tight', dpi=300)
# plt.show()




   

# fig_1 = plt.figure(1, figsize=(6, 5))
# plt.pcolor(ms_x,ms_y,ms_ysol, cmap='jet', shading='auto')
# h=plt.colorbar()
# h.ax.tick_params(labelsize=20)
# plt.xticks([])
# plt.yticks([])
# plt.savefig('NNsolution',bbox_inches='tight')
# #plt.show()



fig_2 = plt.figure(2, figsize=(6, 5))
plt.pcolor(ms_x,ms_y,ms_ugt, cmap='jet', shading='auto')
h=plt.colorbar()
h.ax.tick_params(labelsize=20)
plt.xticks([])
plt.yticks([])
plt.savefig('GTsolution',bbox_inches='tight')
#plt.show()

fig_3 = plt.figure(3, figsize=(6, 5))
plt.pcolor(ms_x,ms_y,abs(ms_ugt-ms_ysol), cmap='jet', shading='auto')
h=plt.colorbar()
h.ax.tick_params(labelsize=20)
plt.xticks([])
plt.yticks([])
plt.savefig('Error',bbox_inches='tight')
#plt.show()









