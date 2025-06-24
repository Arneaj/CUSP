import numpy as np
import matplotlib.pyplot as plt


mu = np.array([[1],[1]])

# theta = np.pi / 4
# sigma = np.array( [ [np.cos(theta), np.sin(theta)], [np.sin(theta), np.cos(theta)] ] )

sigma = np.array( [[1,0], [1,1]] )

E = np.empty((200,200))

sigma_inv = np.linalg.inv(sigma)

for i in range(200):
    for j in range(200):
        X = np.array([[i],[j]]) / 20 - 5
        E[i,j] = (np.transpose(X - mu) @ sigma_inv @ (X-mu))[0,0]
        

mask = np.abs(E - 1) < 1e-1

plt.imshow(mask)
plt.scatter(
    [(mu[0,0]+5)*20], 
    [(mu[1,0]+5)*20]
)

plt.scatter(
    [(mu[0,0]+sigma[1,0]+5)*20, (mu[0,0]+sigma[1,1]+5)*20], 
    [(mu[1,0]+sigma[0,0]+5)*20, (mu[1,0]+sigma[0,1]+5)*20]
)

plt.show()