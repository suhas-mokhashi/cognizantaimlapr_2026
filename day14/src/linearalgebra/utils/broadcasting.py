#create 2 matrix
import numpy as np
#matrix a of shape (3,1)
a = np.array([[1],[2],[3]])
#matrix b of shape (1,3)
b = np.array([[4,5,6]])
#broadcasting a and b to shape (3,3)
c = a + b
print(c)
