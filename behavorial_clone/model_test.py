import numpy as np
import torch

def dec2bin(val, n):
    output = np.zeros((1,n))
    for i in np.flip(np.arange(n)):
        output[0,i] = (val - 2**i >= 0)
        if (val - 2**i >= 0):
            val -= 2**i

    return np.flip(output)

def select_action(state, action, yaw, pitch):
    val = model(state).max(1).indices.long()
    print(type(val))
    binvec = dec2bin(val, 8).squeeze()
    print(binvec)
    action['forward'] = binvec[4]
    action['left'] = binvec[5]
    action['right'] = binvec[6]
    action['back'] = binvec[7]

    action['camera'] = [5*(binvec[0] - binvec[1]), 5*(binvec[2] - binvec[3])]
    return (action)    

model = torch.load('./mineNet.pt')

model.eval()

X = torch.randint(low = 0, high = 255, size= (1,3,64,64))
X = torch.ones((1,3,64,64))

action = model(X).max(1).indices


print(select_action(X, {}, 0,0))
