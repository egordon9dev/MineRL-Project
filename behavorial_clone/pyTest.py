import collections
import torch
d = collections.OrderedDict()

d['a'] = torch.zeros((1,3,64,64))

print(type(d))