import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# x = torch.empty(5,3)
# print(x)

# x = torch.rand(5,3)
# print(x)

# x = torch.zeros(5,3,dtype = torch.long)
# print(x)

# a = [5.5, 4]
# x = torch.tensor(a)
# print(x)

# x = x.new_ones(5, 3, dtype = torch.double)
# print(x)

# x = torch.randn_like(x, dtype = torch.float)
# print(x)
# print(x.size())

# x = x.view(-1,5)
# print(x.size())

# print(x)
# print(x.numpy())

# a = np.ones((3,3))
# print(a)
# b = torch.from_numpy(a)
# print(b)

# x = torch.ones(7,7)
# print(x)

# x = torch.randn(2,5)
# y = torch.randn(2,3)
# print((torch.cat([x,y], 1)))

# x = torch.tensor([1., 2., 3.], requires_grad=True)
# y = torch.tensor([4., 5., 6.], requires_grad=True)

# z = x+y
# print(z)
# print(z.grad_fn)

# s = z.sum()
# print(s)
# print(s.grad_fn)
# s.backward()
# print(x.grad)

# lin  = nn.Linear(5, 3)
# data = torch.randn(3, 5)
# print(lin(data))

# data = torch.randn(2, 2)
# print(data)
# print(F.relu(data))

# data = torch.randn(3,5)
# print(data)
# print(F.softmax(data, dim=0).sum())

# data = torch.randn(3,5)
# print(torch.sum(data, dim=1))

data = [("me gusta comer en la cafeteria".split(), "SPANISH"),
        ("Give it to me".split(), "ENGLISH"),
        ("No creo que sea una buena idea".split(), "SPANISH"),
        ("No it is not a good idea to get lost at sea".split(), "ENGLISH")]

test_data = [("Yo creo que si".split(), "SPANISH"),
             ("it is lost on me".split(), "ENGLISH")]


wordid = {}

for sent, _ in data+test_data:
	for word in sent:
		if word not in wordid:
			wordid[word] = len(wordid)

print(wordid)

VOCAB_SIZE = len(wordid)
NUM_LABELS = 2

label_to_ix = {'SPANISH': 0, "ENGLISH": 1}

class BoWClassifier(nn.Module):

	def __init__(self, num_labels, vocab_size):
		super(BoWClassifier, self).__init__()
		self.linear = nn.Linear(vocab_size, num_labels)

	def forward(self, bow_vec):
		return F.log_softmax(self.linear(bow_vec), dim = 1)

def make_bow_vector(sentence, wordid):
	vec = torch.zeros(len(wordid))
	for word in sentence:
		vec[wordid[word]] += 1
	return vec.view(1, -1)

def make_target(label, label_to_ix):
	return torch.LongTensor([label_to_ix[label]])

model = BoWClassifier(NUM_LABELS, VOCAB_SIZE)

with torch.no_grad():
	sample = data[0]
	bow_vector = make_bow_vector(sample[0], wordid)
	log_probs = model.forward(bow_vector)
	print(log_probs)

#parameters()是一个迭代类型，包含两个数据，一个是A，一个是bias，linear变换是 y = Ax + b
print(next(model.parameters())[:, wordid["creo"]])

loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in range(100):
	for instance,label in data:
		model.zero_grad()
		bow_vec = make_bow_vector(instance, wordid)
		target = make_target(label, label_to_ix)
		log_probs = model.forward(bow_vec)

		loss = loss_function(log_probs, target)
		loss.backward()
		optimizer.step()

with torch.no_grad():
	for instance, label in test_data:
		bow_vec = make_bow_vector(instance, wordid)
		log_probs = model(bow_vec)
		log_probs = log_probs.numpy()
		print(log_probs`)
		label = np.argmax(log_probs, 1)
		print(label)
