import numpy as np
import matplotlib.pyplot as plt

# v1 = np.random.multivariate_normal(mean=np.array([2, 2]), cov=np.diag([1, 0.5]), size=100)
# c = np.zeros(shape=(100, 3))
# c[range(0,100,3), :] = np.array([255,0,0])
# c[range(1,100,3), :] = np.array([0,0,255])
# c[range(2,100,3), :] = np.array([255,255,0])
# c = np.random.randint(1, 4, size=100)
fig, ax = plt.subplots()
# ns = [3, 50, 3]
# means = [[3,2], [3,2], [1, 1]]
# i=0
# colors = ['blue', 'green', 'orange']
# for color, label in zip(['blue', 'green', 'orange'], ['train set', 'augmented train points', 'test set', ]):
#     n = ns[i]
#     v1 = np.random.multivariate_normal(mean=means[i], cov=np.diag([2, 1]), size=n)
#     ax.scatter(v1[:,0], v1[:,1], c=color, label=label, edgecolors='none')
#     i += 1

v = np.random.multivariate_normal(mean=[2, 2], cov=np.diag([1, 1]), size=40)
ax.scatter(v[:,0], v[:,1], c='blue', label='train set')

for i in range(40):
    v2 = np.random.multivariate_normal(mean=v[i,:], cov=np.diag([0.05, 0.05]), size=10)
    ax.scatter(v2[:,0], v2[:,1], c='green')

ax.scatter(0, 0, c='green', label='augmented points')
v3 = np.random.multivariate_normal(mean=[2, 2], cov=np.diag([1, 1]), size=20)
ax.scatter(v3[:,0], v3[:,1], c='red', label='test set')

ax.legend()
ax.set_ylabel('Saturation')
ax.set_xlabel('Contraste')
ax.xaxis.set_label_position('top')
plt.xticks([], [])
plt.yticks([], [])
plt.show()