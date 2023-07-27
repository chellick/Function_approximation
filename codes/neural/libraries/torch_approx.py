import numpy as np
import matplotlib.pyplot as plt
import torch

x_data = np.arange(1, 100, 1)
y_data = x_data + 2 * np.sin(x_data) ** 2 + 1

x_train = torch.tensor(x_data[:30], dtype=torch.float32)
y_train = torch.tensor(y_data[:30], dtype=torch.float32)

x_train.unsqueeze_(1)
y_train.unsqueeze_(1)

x_test = torch.tensor(x_data[:30], dtype=torch.float32)
y_test = torch.tensor(y_data[:30], dtype=torch.float32)

y_test.unsqueeze_(1)
x_test.unsqueeze_(1)

# train_ds = tf.data.Dataset.from_tensor_slices(
#     (x_train, y_train)).shuffle(10000)

# test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))




class Approx(nn.Module):
  def __init__(self):
    super(Approx, self).__init__()
    self.Layers = nn.Sequential(nn.Linear(1, 512),
                                nn.ReLU(),
                                nn.Linear(512, 512),
                                nn.ReLU(),
                                nn.Linear(512, 1))
  
  def forward(self, x):
    return self.Layers(x)

model = Approx()
loss_fn = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


for epoch in range(3000):
  model.train()
  pred = model(x_train)
  loss = loss_fn(pred, y_train)
  print(f'loss => {loss.item()}')
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

pred = model(x_test)

plt.scatter(x_test, y_test, label='True', c='b')
plt.scatter(x_test, pred.detach().numpy(), label='Pred', c='r')
plt.show()
