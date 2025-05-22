import numpy as np

a=np.load('../data/edges.npy',allow_pickle=True)
b = [[], []]
for i in range(len(a)):
    b[i]=str(list(a[i]))
with open('./edges.txt','w') as f:
    f.write('\n'.join(b))
