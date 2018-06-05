import csv
import numpy as np
import matplotlib.pyplot as plt
import os

filename = 'train.csv'
dataset = []
labelset = []

f = open(filename, 'r')
lines = f.readlines()
for i in range(1, len(lines)):
	datas = lines[i].split(',')
	for i in range(len(datas)):
		datas[i] = int(datas[i])
	dataset.append(np.array(datas[1:785]))
	labelset.append(datas[0])
f.close()
print('finish dataload')
	
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(1, 6, 5)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(6, 16, 3)
		self.fc1 = nn.Linear(16 * 5 * 5, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(-1, 16 * 5 * 5)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
#os.environ['CUDA_VISIBLE_DEVICES'] = 0
net = Net()
#net = net.cuda()
net.to(device)
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

axisx = []
axisytrain = []
axisytest = []

for epoch in range(20):  # loop over the dataset multiple times

	running_loss = 0.0
	for i in range(0, len(dataset)-6000, 10):
		# get the inputs
		input = np.array(dataset[i:i+10])
		input = input.reshape((10, 1, 28, 28)).astype('float32')
		label = np.array(labelset[i:i+10])
		inputs = torch.from_numpy(input)
		labels = torch.LongTensor(label)
		inputs, labels = inputs.to(device), labels.to(device)
		optimizer.zero_grad()
		outputs = net(inputs)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

		# print statistics
		running_loss += loss.item()
		if i % 200 == 1999:    # print every 2000 mini-batches
			print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 2000))
			running_loss = 0.0
	axisx.append(epoch)		
	cnt = 0
	cor = 0
	for i in range(0, len(dataset)-6000):
		# get the inputs
		input = dataset[i]
		input = input.reshape((1, 1, 28, 28)).astype('float32')
		label = np.array([labelset[i]]).astype('float32')
		inputs = torch.from_numpy(input)
		labels = torch.LongTensor(label)
		inputs, labels = inputs.to(device), labels.to(device)
		# zero the parameter gradients
		optimizer.zero_grad()

		# forward + backward + optimize
		outputs = net(inputs)
		_, predicted = torch.max(outputs, 1)
		if predicted == labels[0]:
			cor = cor + 1
		cnt = cnt + 1
	axisytrain.append(cor/cnt)
	cnt = 0
	cor = 0
	for i in range(len(dataset)-6000, len(dataset)):
		# get the inputs
		input = dataset[i]
		input = input.reshape((1, 1, 28, 28)).astype('float32')
		label = np.array([labelset[i]]).astype('float32')
		inputs = torch.from_numpy(input)
		labels = torch.LongTensor(label)
		inputs, labels = inputs.to(device), labels.to(device)
		# zero the parameter gradients
		optimizer.zero_grad()

		# forward + backward + optimize
		outputs = net(inputs)
		_, predicted = torch.max(outputs, 1)
		if predicted == labels[0]:
			cor = cor + 1
		cnt = cnt + 1
	axisytest.append(cor/cnt)
	print('train: %f\n test: %f'%(axisytrain[epoch], axisytest[epoch]))

print('Finished Training')
plt.plot(axisx, axisytrain)
plt.plot(axisx, axisytest)
plt.show()

'''
filename = 'test.csv'
testset = []
resultset = []

f = open(filename, 'r')
lines = f.readlines()
for i in range(1, len(lines)):
	datas = lines[i].split(',')
	for i in range(len(datas)):
		datas[i] = int(datas[i])
	testset.append(np.array(datas[0:784]))
f.close()
print('finish testdataload')

for i in range(len(testset)):
	# get the inputs
	input = testset[i]
	input = input.reshape((1, 1, 28, 28)).astype('float32')
	inputs = torch.from_numpy(input)
	#print(inputs)
	#print(labels)
	# zero the parameter gradients
	optimizer.zero_grad()

	# forward + backward + optimize
	outputs = net(inputs)
	_, predicted = torch.max(outputs, 1)
	
	resultset.append(predicted)

filename = 'submission.csv'
f = open(filename, 'w')
f.write('ImageId,Label\n')
for i in range(len(resultset)):
	f.write('%d,%d\n'%(i+1, resultset[i]))
f.close()
'''
	