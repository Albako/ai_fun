import torch
from torch import nn
import matplotlib.pyplot as plt


weight = 1.0
bias = 2.1


start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias


train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]


def plot_predictions(train_data=X_train,
                     train_labels=y_train,
                     test_data=X_test,
                     test_labels=y_test,
                     predictions=None):
    plt.figure(figsize=(10, 7))
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")
    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")
    plt.legend(prop={"size": 14})
    plt.show()


class LinearRegreModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float)) 
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float)) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias

torch.manual_seed(42)
model_01 = LinearRegreModel()


loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model_01.parameters(), lr=0.001)

# Trenowanie modelu dopóki strata nie osiągnie 0.001
target_loss = 0.001
current_loss = float('inf')  # Ustawienie wysokiej początkowej wartości straty
epoch = 0

while current_loss > target_loss:
    model_01.train()
    
    y_pred = model_01(X_train)
    loss = loss_fn(y_pred, y_train)
    current_loss = loss.item()
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 5 == 0:
        print(f"Epoch {epoch}: Loss = {current_loss}")
    
    epoch += 1

print(f"Training completed in {epoch} epochs with final loss = {current_loss:.4f}")
with torch.inference_mode():
    y_preds = model_01(X_test)
    plot_predictions(predictions=y_preds)
