
import torch
import torch.nn as nn
# NLP Example
# batch, sentence_length, embedding_dim = 3,2,5
# embedding = torch.ones(batch, sentence_length, embedding_dim)
# embedding[:,:,1] = 2
# embedding[:,:,2] = 3
# embedding[:,:,3] = 4
# embedding[:,:,4] = 5
# layer_norm = nn.LayerNorm(embedding_dim)
# # Activate module
# output = layer_norm(embedding)
# layer_norm(torch.concat([embedding, embedding], dim=1))
# breakpoint()
# # Image Example
# N, C, H, W = 20, 5, 10, 10
# input = torch.randn(N, C, H, W)
# # Normalize over the last three dimensions (i.e. the channel and spatial dimensions)
# # as shown in the image below
# layer_norm = nn.LayerNorm([C, H, W])
# output = layer_norm(input)


# With square kernels and equal stride
m = nn.Conv2d(16, 33, 3, stride=2)
# non-square kernels and unequal stride and with padding
m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
# non-square kernels and unequal stride and with padding and dilation
m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
input = torch.randn(20, 16, 50, 100)
output = m(input)
breakpoint()