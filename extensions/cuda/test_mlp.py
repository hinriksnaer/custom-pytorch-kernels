from models import CustomCUDAMLP
import torch

x = torch.randn(16, 32, device="cuda", requires_grad=True)
y = torch.randn(16, 64, device="cuda")

mlp = CustomCUDAMLP(32, 64, 64).cuda()
opt = torch.optim.SGD(mlp.parameters(), lr=0.01)

for i in range(10):
    opt.zero_grad()
    out = mlp(x)
    loss = ((out - y) ** 2).mean()
    loss.backward()
    opt.step()
    print(f"Step {i}, loss = {loss.item():.4f}")
