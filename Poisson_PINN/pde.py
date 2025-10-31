import torch

mse_loss = torch.nn.MSELoss()

def pde(x1,x2,net):
    out = net(x1,x2)
    u_x = torch.autograd.grad(out.sum(),x1,create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(),x1, create_graph=True)[0]

    u_y = torch.autograd.grad(out.sum(),x2,create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y.sum(),x2,create_graph=True)[0]

    return -u_xx-u_yy


def bdry(x1,x2,net):
    out = net(x1,x2)
    return out


def pdeloss(net,intx1,intx2,pdedata,bdx1,bdx2,bdrydata,bw):
    pout = pde(intx1,intx2,net)
    bout = bdry(bdx1,bdx2,net)
    pres = mse_loss(pout,pdedata)
    bres = mse_loss(bout,bdrydata)

    loss = pres + bw*bres

    return loss, pres, bres 

