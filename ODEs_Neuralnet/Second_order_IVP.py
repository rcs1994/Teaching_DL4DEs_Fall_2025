import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(0)
np.random.seed(0)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Neural network (fully connected)
# -------------------------
class PINN(nn.Module):
    def __init__(self, hidden_layers=(40,40,40), activation=nn.Tanh()):
        super().__init__()
        layers = []
        in_dim = 1
        for h in hidden_layers:
            layers.append(nn.Linear(in_dim, h))
            layers.append(activation)
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, t):
        # expects t shape (N,1)
        return self.net(t)

# -------------------------
# Exact solution for comparison (derived analytically)
# y'' + 3 y' + 2 y = t^2, y(0)=1, y'(0)=0
# Solution: y(t) = -0.75 e^{-2t} + 0.5 t^2 - 1.5 t + 1.75
# -------------------------
def y_exact(t):
    return -0.75 * np.exp(-2.0 * t) + 0.5 * t**2 - 1.5 * t + 1.75

# -------------------------
# collocation points in domain [0, T1]
# -------------------------
T0 = 0.0
T1 = 2.0
n_collocation = 100
t_collocation = np.linspace(T0, T1, n_collocation).reshape(-1, 1).astype(np.float32)
t_collocation_tensor = torch.tensor(t_collocation, requires_grad=True, device=device)

# initial condition point(s)
t0 = torch.tensor([[0.0]], requires_grad=True, device=device)
y0_value = torch.tensor([[1.0]], device=device)
yp0_value = torch.tensor([[0.0]], device=device)  # y'(0) = 0

# model
model = PINN(hidden_layers=(40,40,40)).to(device)

# optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-3)
mse_loss = nn.MSELoss(reduction='mean')

# -------------------------
# training loop
# -------------------------
n_epochs = 8000
print_every = 500

for epoch in range(1, n_epochs+1):
    model.train()
    optimizer.zero_grad()

    # collocation input (ensure requires_grad True for derivatives)
    t_in = t_collocation_tensor.clone().detach().requires_grad_(True)

    # forward
    y_pred = model(t_in)                       # shape (N,1)

    # first derivative dy/dt
    dy_dt = torch.autograd.grad(
        y_pred, t_in,
        grad_outputs=torch.ones_like(y_pred),
        create_graph=True,
        retain_graph=True
    )[0]

    # second derivative d2y/dt2
    d2y_dt2 = torch.autograd.grad(
        dy_dt, t_in,
        grad_outputs=torch.ones_like(dy_dt),
        create_graph=True,
        retain_graph=True
    )[0]

    # PDE residual: y'' + 3 y' + 2 y - t^2
    t_sq = t_in ** 2.0
    residual = d2y_dt2 + 3.0 * dy_dt + 2.0 * y_pred - t_sq

    # residual loss (use MSE over collocation points)
    loss_res = mse_loss(residual, torch.zeros_like(residual)) * t_in.shape[0]  # scale to sum-like

    # initial condition losses
    y0_pred = model(t0)
    loss_ic_y = mse_loss(y0_pred, y0_value)  # (y(0)-1)^2 averaged

    # compute y'(0) by autograd
    t0_for_grad = t0.clone().detach().requires_grad_(True)
    y0_for_grad = model(t0_for_grad)
    dy0_pred = torch.autograd.grad(
        y0_for_grad, t0_for_grad,
        grad_outputs=torch.ones_like(y0_for_grad),
        create_graph=True
    )[0]
    loss_ic_yp = mse_loss(dy0_pred, yp0_value)  # (y'(0)-0)^2 averaged

    # total loss (weights can be tuned)
    loss = loss_res + 100.0 * loss_ic_y + 100.0 * loss_ic_yp
    # note: I multiplied IC losses by 100 to ensure good enforcement; adjust if needed.

    loss.backward()
    optimizer.step()

    if epoch % print_every == 0 or epoch == 1:
        model.eval()
        with torch.no_grad():
            t_test = np.linspace(T0, T1, 400).reshape(-1,1).astype(np.float32)
            t_test_t = torch.tensor(t_test, device=device)
            y_pred_test = model(t_test_t).cpu().numpy().flatten()
            y_ex = y_exact(t_test.flatten())
            l2_rel = np.linalg.norm(y_pred_test - y_ex) / (np.linalg.norm(y_ex) + 1e-12)
        print(f"Epoch {epoch:5d} | Loss = {loss.item():.4e} | Loss_res = {loss_res.item():.4e} | "
              f"IC_y = {loss_ic_y.item():.4e} | IC_yp = {loss_ic_yp.item():.4e} | Rel L2 = {l2_rel:.4e}")

# -------------------------
# After training: evaluate & plot
# -------------------------
model.eval()
t_plot = np.linspace(T0, T1, 400).reshape(-1,1).astype(np.float32)
t_plot_t = torch.tensor(t_plot, device=device)
with torch.no_grad():
    y_pred_plot = model(t_plot_t).cpu().numpy().flatten()

t_plot_flat = t_plot.flatten()
y_ex_plot = y_exact(t_plot_flat)

abs_err = np.abs(y_pred_plot - y_ex_plot)
max_err = np.max(abs_err)
l2_err = np.linalg.norm(y_pred_plot - y_ex_plot) / np.sqrt(len(y_ex_plot))
print(f"\nFinal errors: L-inf = {max_err:.4e}, L2 (rms) = {l2_err:.4e}")

# Plot: predicted vs exact and absolute error
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(t_plot_flat, y_ex_plot, label='Exact y(t)', linewidth=2)
plt.plot(t_plot_flat, y_pred_plot, '--', label='PINN y_theta(t)')
plt.scatter(t_collocation.flatten(), np.zeros_like(t_collocation.flatten()), s=6,
            label='collocation points (x-axis)', alpha=0.25)
plt.xlabel('t')
plt.ylabel('y')
plt.legend()
plt.title('PINN solution vs exact')

plt.subplot(1,2,2)
plt.plot(t_plot_flat, abs_err)
plt.xlabel('t')
plt.ylabel('|y_pred - y_exact|')
plt.title('Absolute error')

plt.tight_layout()
plt.show()

# -------------------------
# (Optional) -- your original first-order example for dy/dt = sin(t) is simpler:
# residual = dy_dt - sin(t); IC: y(0)=1
# The above structure generalizes to second-order by adding the second derivative and the additional IC for y'(0).
# -------------------------
