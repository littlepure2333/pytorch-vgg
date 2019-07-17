import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

images_path = "./HCL2000-100/HCL2000_100_train.npz"
labels_path = "./HCL2000-100/HCL2000_100_train_label.npz"
images_data = np.load(images_path)
labels_data = np.load(labels_path)
print(images_data)
print(labels_data)
images = images_data["arr_0"]
labels = labels_data["arr_0"]
print(images.shape)
print(labels.shape)

# 1张图里显示4张子图，r代表reverse
plt.figure()
plt.subplot(221)
plt.imshow(images[0].reshape((28, 28)), cmap=plt.cm.gray)
plt.subplot(222)
plt.imshow(images[1].reshape((28, 28)), cmap=plt.cm.gray_r)
plt.subplot(223)
plt.imshow(images[2].reshape((28, 28)), cmap='gray')
plt.subplot(224)
pilimage = Image.fromarray(images[3].reshape((28, 28)))
plt.imshow(pilimage, cmap='gray_r')
# plt.imshow(images[3].reshape((28, 28)), cmap='gray_r')
plt.show()

# 分别显示4张图
for i in (x + 4 for x in range(4)):
    plt.figure()
    plt.imshow(images[i].reshape((28, 28)), cmap=plt.cm.gray)
    print('id:', i)
    print('label:', labels[i])
    print('size:', images[i].size)
    plt.show()
