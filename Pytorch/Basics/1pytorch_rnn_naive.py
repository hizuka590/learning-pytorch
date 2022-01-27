"""
Example code of a simple RNN, naive implementation

Programmed by hizuka <duanxin@connect.hku.hk>
*    2020-05-09 Initial coding

"""

# Imports
import torch
# import torchvision  # torch package for vision related things
# import torch.nn.functional as F  # Parameterless functions, like (some) activation functions
import torchvision.datasets as datasets  # Standard datasets
import torchvision.transforms as transforms  # Transformations we can perform on our dataset for augmentation
from matplotlib import pyplot as plt
from torch import optim  # For optimizers like SGD, Adam, etc.
from torch import nn  # All neural network modules
from torch.utils.data import DataLoader  # Gives easier dataset managment by creating mini batches etc.
from tqdm import tqdm  # For a nice progress bar!

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
input_size = 784
hidden_size = 256
num_classes = 10
learning_rate = 0.0005
batch_size = 64
num_epochs = 300
pilot_steps = 100

# Recurrent neural network (many-to-one)
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        # self.num_layers = num_layers
        # self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        # self.fc = nn.Linear(hidden_size * sequence_length, num_classes)
        self.i2h = nn.Linear(input_size+hidden_size,hidden_size)
        self.i2o = nn.Linear(input_size+hidden_size,num_classes)
        self.softmax = nn.LogSoftmax(dim = 1)


    def forward(self, input_tensor, hiddent_tensor):
        # print("forward fuction===: input_tensor.shape, hidden_tensor.shape",input_tensor.shape, hiddent_tensor.shape)
        combined = torch.cat((input_tensor,hiddent_tensor),1)
        # print("combined tensor.shape: ",combined.shape)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output,hidden

        # # Forward propagate LSTM
        # out, _ = self.rnn(x, h0)
        # out = out.reshape(out.shape[0], -1)
        #
        # # Decode the hidden state of the last time step
        # out = self.fc(out)
        # return out
    def init_hidden(self,batch_size):
        return torch.ones(1,self.hidden_size)


# Load Data
train_dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=False)
test_dataset = datasets.MNIST(root="dataset/", train=False, transform=transforms.ToTensor(), download=False)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Initialize network (try out just using simple RNN, or GRU, and then compare with LSTM)
model = RNN(input_size, hidden_size, num_classes).to(device)
# one step sanity check
# for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
#     # Get data to cuda if possible
#     data = data.to(device=device).squeeze(1)
#     targets = targets.to(device=device)
#     input_tensor = data.reshape(batch_size,-1).to(device)
#     print("input shape:", input_tensor.shape)
#     hidden_tensor = model.init_hidden(batch_size).to(device)
#     print("hidden0 shape:", hidden_tensor.shape)
#     output, next_hidden = model(input_tensor, hidden_tensor)
#     print("output shape:", output.shape)
#     print("next hidden shape:", next_hidden.shape)
#     break

# # Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#
# # Train Network
def train(input,output_y):
    hidden_tensor = model.init_hidden(batch_size)
    # print("train=====input.shape hidden.shape output.shape", input.shape,hidden_tensor.shape, output_y.shape)
    hidden_tensor = hidden_tensor.to(device)
    output = torch.zeros(input.shape[0])
    for i in range(input.shape[0]):
        output[i],hidden = model(input_tensor[i].reshape(1,-1),hidden_tensor)
    loss = criterion(output , output_y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return output, loss.item()

current_loss = 0
losses = []
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
        # Get data to cuda if possible
        data = data.to(device=device).squeeze(1)
        targets = targets.to(device=device)
        input_tensor = data.reshape(batch_size, -1).to(device)

        output, loss = train(input_tensor, targets)
        current_loss += loss
        if (epoch+1)%pilot_steps == 0:
            losses.append(current_loss / pilot_steps) #average
            current_loss = 0
        # break

plt.figure()
plt.plot(losses)
plt.show()


#
# # Check accuracy on training & test to see how good our model
# def check_accuracy(loader, model):
#     num_correct = 0
#     num_samples = 0
#
#     # Set model to eval
#     model.eval()
#
#     with torch.no_grad():
#         for x, y in loader:
#             x = x.to(device=device).squeeze(1)
#             y = y.to(device=device)
#
#             scores = model(x)
#             _, predictions = scores.max(1)
#             num_correct += (predictions == y).sum()
#             num_samples += predictions.size(0)
#
#     # Toggle model back to train
#     model.train()
#     return num_correct / num_samples
#
#
# print(f"Accuracy on training set: {check_accuracy(train_loader, model)*100:2f}")
# print(f"Accuracy on test set: {check_accuracy(test_loader, model)*100:.2f}")
