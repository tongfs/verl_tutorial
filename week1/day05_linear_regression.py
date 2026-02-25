import torch
import torch.nn as nn

# 1. 构造数据：y = 2x + 1 + 噪声
torch.manual_seed(42)
x = torch.randn(100, 1) * 3
y = 2 * x + 1 + torch.randn(100, 1) * 0.5

# 2. 定义模型
model = nn.Linear(1, 1)
print("训练前:", model.weight.item(), model.bias.item())

# 3. 损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 4. 训练循环
for epoch in range(100):
    optimizer.zero_grad()
    pred = model(x)
    loss = criterion(pred, y)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

print("训练后:", model.weight.item(), model.bias.item())
# 应接近 2 和 1
