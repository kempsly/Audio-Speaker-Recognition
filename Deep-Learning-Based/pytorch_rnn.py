"""
    Generate random sequences
"""

import numpy as np
import pickle
import random

n_train = 10000
n_test = 100

train_samples = []
train_labels = []
test_samples = []
test_labels = []

for i in range(n_train + n_test):
  if i < n_train:
    samples = train_samples
    labels = train_labels
  else:
    samples = test_samples
    labels = test_labels

  label = random.choice([0, 1])
  start = random.uniform(-40.0, 40.0)
  sample = [start]
  next = start + random.uniform(-1, 5.0) * (-label * 2 + 1)
  sample.append(next)
  next = next + random.uniform(-2, 10.0) * (-label * 2 + 1)
  sample.append(next)
  next = next + random.uniform(-2, 10.0) * (-label * 2 + 1)
  sample.append(next)


  print("label:", label)
  print("sample:", sample)
  labels.append(label)
  samples.append(sample)

  data = {
    "train_samples": train_samples,
    "train_labels": train_labels,
    "test_samples": test_samples,
    "test_labels": test_labels,
  }

with open('data.pickle', 'wb') as f:
  pickle.dump(data, f)
 
  
"""
    Build the recurrent neural network
"""
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim


def GetData():
  with open('data.pickle', mode='rb') as f:
    data = pickle.load(f)

  train_samples = np.array(data['train_samples'])
  train_labels = np.array(data['train_labels'])
  test_samples = np.array(data['test_samples'])
  test_labels = np.array(data['test_labels'])
  return train_samples, train_labels, test_samples, test_labels

class MyNet(nn.Module):
  def __init__(self):
    super(MyNet, self).__init__()
    # Define the RNN layer and a final linear layer
    self.rnn = nn.RNN(
      input_size=1,
      hidden_size=8,
      num_layers=1)
    self.feedforward_output = nn.Linear(
      in_features=8,
      out_features=1)

  def forward(self, x):
    h0 = torch.zeros(1, x.shape[1], 8)
    y, hn = self.rnn(x, h0)
    y = torch.sigmoid(self.feedforward_output(y))
    return y


def main():
  train_samples, train_labels, test_samples, test_labels = GetData()

  mynet = MyNet()

  # Train
  criterion = nn.BCELoss()
  optimizer = optim.Adam(mynet.parameters(), lr=0.1)
  num_steps = 100
  print('start training')
  for epoch in range(num_steps):
      optimizer.zero_grad()
      inputs = np.expand_dims(train_samples.transpose(), -1)
      outputs = torch.squeeze(
          mynet(torch.from_numpy(inputs).float()))
      last_outputs = outputs[-1, :]
      loss = criterion(
          last_outputs, torch.from_numpy(train_labels).float())
      loss.backward()
      optimizer.step()
      print('epoch:', epoch, 'loss:', loss.item())
  print('finished training')

  # Evaluate
  inputs = np.expand_dims(test_samples.transpose(), -1)
  test_predict = np.squeeze(mynet(torch.from_numpy(inputs).float()).data.numpy())
  last_predict = (test_predict[-1, :] > 0.5)
  accuracy = sum(last_predict == test_labels) / len(test_labels)
  print('accuracy on test data:', accuracy)

if __name__ == '__main__':
  main()