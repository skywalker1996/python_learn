import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

CONTEXT_SIZE = 2
EMBEDDING_DIM = 10


test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()


trigrams = [([test_sentence[i], test_sentence[i]], test_sentence[i+2]) for i in range(len(test_sentence)-2)]

print(trigrams[:3])

vocab = set(test_sentence)
wordid = {word: i for i, word in enumerate(vocab)}

class NGramLanguageModeler(nn.Module):

	#context是指x的单词个数
	def __init__(self, vocab_size, embedding_dim, context_size):
		super(NGramLanguageModeler, self).__init__()
		self.embeddings = nn.Embedding(vocab_size, embedding_dim)
		self.linear1 = nn.Linear(context_size * embedding_dim, 128)
		self.linear2 = nn.Linear(128, vocab_size)

	def forward(self, inputs):
		embeds = self.embeddings(inputs).view((1, -1))
		out = F.relu(self.linear1(embeds))
		out = self.linear2(out)
		log_probs = F.log_softmax(out, dim=1)
		return log_probs

losses = []
loss_function = nn.NLLLoss()
model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr = 0.001)

for epoch in range(1000):
	total_loss = torch.Tensor([0])
	for context, target in trigrams:
		context_idxs = torch.tensor([wordid[w] for w in context], dtype = torch.long)
		model.zero_grad()
		log_probs = model(context_idxs)
		loss = loss_function(log_probs, torch.tensor([wordid[target]], dtype = torch.long))
		loss.backward()
		optimizer.step()
		total_loss+=loss.item()
	#losses.append(total_loss)
	print(total_loss)
#print(losses)

