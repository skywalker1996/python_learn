import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

CONTEXT_SIZE = 2
EMBEDDING_DIM = 10

raw_text = """When forty winters shall besiege thy brow,
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

vocab = set(raw_text)
vocab_size = len(vocab)

wordid = {word:i for i, word in enumerate(vocab)}

data = []

for i in range(2, len(raw_text)-2):
	context = [raw_text[i-2], raw_text[i-1], raw_text[i+1], raw_text[i+2]]
	target = raw_text[i]
	data.append((context, target))

print(data[:5])


class CBOW(nn.Module):
	def __init__(self, vocab_size, embedding_dim, context_size):
		super(CBOW, self).__init__()
		self.embeddings = nn.Embedding(vocab_size, embedding_dim)
		self.linear1 = nn.Linear(context_size*2*embedding_dim, 128)
		self.linear2 = nn.Linear(128, vocab_size)

	def forward(self, inputs):
		embeds = self.embeddings(inputs).view(1, -1)
		out = F.relu(self.linear1(embeds))
		out = self.linear2(out)
		log_probs = F.log_softmax(out, dim = 1)
		return log_probs


def make_context_vector(context, wordid):
	idxs = [wordid[w] for w in context]
	return torch.tensor(idxs, dtype=torch.long)



losses = []
loss_function = nn.NLLLoss()
model = CBOW(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr = 0.001)

for epoch in range(1000):
	total_loss = torch.Tensor([0])
	for context, target in data:
		context_idxs = make_context_vector(context, wordid)
		model.zero_grad()
		log_probs = model(context_idxs)
		loss = loss_function(log_probs, torch.tensor([wordid[target]], dtype = torch.long))
		loss.backward()
		optimizer.step()
		total_loss+=loss.item()
	#losses.append(total_loss)
	print(total_loss)
#print(losses)