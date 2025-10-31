"""
PINN for dy/dt = sin(t), y(0)=1
Exact solution: y(t) = -cos(t) + 2
Loss: sum_i |d y_theta/dt (t_i) - sin(t_i)|^2 + |y_theta(0) - 1|^2
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
torch.manual_seed(0)
np.random.seed(0)

from time import time

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Neural network (fully connected)
class PINN(nn.Module):
    def __init__(self, hidden_layers=(20,20,20), activation=nn.Tanh()):
        super().__init__()
        layers = []
        in_dim = 1
        for h in hidden_layers:
            layers.append(nn.Linear(in_dim, h))
            layers.append(activation)
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))  #last layer it is 50 to 1
        self.net = nn.Sequential(*layers)

    def forward(self, t):
        # expects t shape (N,1)
        return self.net(t)

# exact solution for comparison
def y_exact(t):
    return -np.cos(t) + 2.0

# collocation points in domain [0, 2*pi]
T0 = 0.0
#T1 = 2.0 * np.pi
T1 = 2.0
n_collocation = 20  # number of collocation points

t_collocation = np.linspace(T0, T1, n_collocation).reshape(-1, 1).astype(np.float32)
t_collocation_tensor = torch.tensor(t_collocation, requires_grad=True, device=device)

# initial condition point
t0 = torch.tensor([[0.0]], requires_grad=True, device=device)
y0_value = torch.tensor([[1.0]], device=device)

# model
model = PINN(hidden_layers=(20,20,20)).to(device)

# optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-3)
# optional LBFGS for a few steps later
use_lbfgs = True

mse_loss = nn.MSELoss()

# training loop
n_epochs = 5000
print_every = 500

start_time = time()

for epoch in range(1, n_epochs+1):
    model.train()
    optimizer.zero_grad()

    # predict at collocation points
    t_in = t_collocation_tensor.clone().detach().requires_grad_(True)
    y_pred = model(t_in)                     # shape (N,1)

    # compute dy/dt using autograd
    dy_dt = torch.autograd.grad(y_pred, t_in, grad_outputs=torch.ones_like(y_pred), create_graph=True)[0]

    # residual: dy/dt - sin(t)
    sin_t = torch.sin(t_in)
    res = dy_dt - sin_t

    loss_res = mse_loss(res, torch.zeros_like(res))     # sum of squared residuals via MSE (mean) - scaled
    # To match user-specified sum rather than mean, scale by number of points:
    loss_res = loss_res * t_in.shape[0]

    # initial condition penalty (y(0) - 1)^2
    y0_pred = model(t0)
    loss_ic = mse_loss(y0_pred, y0_value) * 1.0  # multiply by 1 (just clarity)

    # total loss
    loss = loss_res + loss_ic

    loss.backward()
    optimizer.step()
    
    
   

    # # Optionally run LBFGS occasionally for refinement
    # if use_lbfgs and epoch % 500 == 0:
    #     def closure():
    #         optimizer_lbfgs.zero_grad()
    #         t_in2 = t_collocation_tensor.clone().detach().requires_grad_(True)
    #         y_pred2 = model(t_in2)
    #         dy_dt2 = torch.autograd.grad(y_pred2, t_in2, grad_outputs=torch.ones_like(y_pred2), create_graph=True)[0]
    #         res2 = dy_dt2 - torch.sin(t_in2)
    #         loss_res2 = mse_loss(res2, torch.zeros_like(res2)) * t_in2.shape[0]
    #         y0_pred2 = model(t0)
    #         loss_ic2 = mse_loss(y0_pred2, y0_value)
    #         loss_tot = loss_res2 + loss_ic2
    #         loss_tot.backward()
    #         return loss_tot

    #     # create a small LBFGS optimizer on the current params
    #     optimizer_lbfgs = optim.LBFGS(model.parameters(), max_iter=100, tolerance_grad=1e-7, tolerance_change=1e-9)
    #     optimizer_lbfgs.step(closure)

    if epoch % print_every == 0 or epoch == 1:
        # compute current L2 error on a fine grid
        model.eval()
        with torch.no_grad():
            t_test = np.linspace(T0, T1, 400).reshape(-1,1).astype(np.float32)
            t_test_t = torch.tensor(t_test, device=device)
            y_pred_test = model(t_test_t).cpu().numpy().flatten()
            y_ex = y_exact(t_test.flatten())
            l2_rel = np.linalg.norm(y_pred_test - y_ex) / (np.linalg.norm(y_ex) + 1e-12)
        print(f"Epoch {epoch:5d} | Loss = {loss.item():.4e} | Loss_res = {loss_res.item():.4e} | Loss_ic = {loss_ic.item():.4e} | Rel L2 = {l2_rel:.4e}")

# After training: evaluate & plot


end_time = time()
print(f"Training time: {end_time - start_time:.2f} seconds") 


model.eval()
t_plot = np.linspace(T0, T1, 400).reshape(-1,1).astype(np.float32)
t_plot_t = torch.tensor(t_plot, device=device)
with torch.no_grad():
    y_pred_plot = model(t_plot_t).cpu().numpy().flatten()

t_plot_flat = t_plot.flatten()
y_ex_plot = y_exact(t_plot_flat)

# compute errors
abs_err = np.abs(y_pred_plot - y_ex_plot)
max_err = np.max(abs_err)
l2_err = np.linalg.norm(y_pred_plot - y_ex_plot) / np.sqrt(len(y_ex_plot))

print(f"\nFinal errors: L-inf = {max_err:.4e}, L2 (rms) = {l2_err:.4e}")

# Plot: predicted vs exact and absolute error
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(t_plot_flat, y_ex_plot, label='Exact y(t)', linewidth=2)
plt.plot(t_plot_flat, y_pred_plot, '--', label='PINN y_theta(t)')
plt.scatter(t_collocation.flatten(), np.zeros_like(t_collocation.flatten()), s=4, label='collocation points', alpha=0.25)  # just show collocation locations on x-axis
plt.xlabel('t')
plt.ylabel('y')
plt.legend()
plt.title('PINN solution vs exact')

plt.subplot(1,2,2)
plt.plot(t_plot_flat, abs_err)
plt.xlabel('t')
plt.ylabel('|y_pred - y_exact|')
plt.title('Absolute error (pinpoint)')

plt.tight_layout()
plt.show()




# # plot exact solution in its own figure
# plt.figure(figsize=(8,4))
# plt.plot(t_plot_flat, y_ex_plot, label='Exact y(t)', linewidth=2)
# plt.xlabel('t')
# plt.ylabel('y')
# plt.title('Exact solution')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()

# # plot PINN (computed) solution in its own figure
# plt.figure(figsize=(8,4))
# plt.plot(t_plot_flat, y_pred_plot, '--', label='PINN y_theta(t)')
# plt.scatter(t_collocation.flatten(), np.zeros_like(t_collocation.flatten()), s=12,
#             label='collocation points (x-axis)', alpha=0.25)
# plt.xlabel('t')
# plt.ylabel('y')
# plt.title('PINN computed solution')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()

# # plot absolute error in its own figure
# plt.figure(figsize=(8,4))
# plt.plot(t_plot_flat, abs_err, label='|y_pred - y_exact|')
# plt.xlabel('t')
# plt.ylabel('Absolute error')
# plt.title('Absolute error')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()

# show all figures
#plt.show()

