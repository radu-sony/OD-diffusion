import torch
import torch.nn as nn
import time
device = torch.device('cuda:7')  # GPU 1 is 'cuda:1'
torch.cuda.set_device(device)

# Define a simple neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 5)  # 10 input features, 5 output features

    def forward(self, x):
        return self.fc1(x)

# Create an instance of the network
model = SimpleNet()

# Check if GPU is available and move the model to GPU
if torch.cuda.is_available():
    model.cuda()
    print("Model moved to GPU.")
else:
    print("GPU is not available.")

# Create a dummy input tensor and move it to GPU if available
input_tensor = torch.randn(1, 10)  # batch size of 1, 10 input features
if torch.cuda.is_available():
    input_tensor = input_tensor.cuda()



# Perform a forward pass
model.train()
output = model(input_tensor)
print(output)
time.sleep(10)
