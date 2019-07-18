from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms

"""
the beginning images and labels are ndarray,
after init images are still ndarray, labels are tensor.
should be transformed to PIL image eventually to Tensor
"""


class HCL(Dataset):
    def __init__(self, images_path, labels_path, transform=None, target_transform=None):
        images_data = np.load(images_path)
        labels_data = np.load(labels_path)
        images_array = images_data["arr_0"]  # N*(H*W)
        images_array_reshape = images_array.reshape((images_array.shape[0], 28, 28))  # N*H*W
        images = np.expand_dims(images_array_reshape, axis=1)  # N*C*H*W
        self.images = images.transpose((0, 2, 3, 1))  # N*H*W*C
        labels = labels_data["arr_0"]
        labels_map = label_map(labels)
        self.labels = torch.from_numpy(labels_map.astype(np.long))
        self.transforms = transform
        self.target_transforms = target_transform

    def __getitem__(self, index):
        image = self.images[index]  # H*W*C
        label = self.labels[index]
        if self.transforms is not None:
            image = self.transforms(image)
        if self.target_transforms is not None:
            label = self.target_transforms(label)
        # If the transform variable is not empty
        # then it applies the operations in the transforms with the order that it is created.
        return image, label

    def __len__(self):
        return self.images.shape[0]  # of how many data(images?) you have


# 把原来的label映射到0-99
def label_map(labels):
    label_dict = {}
    current_index = 0
    map_index = 0
    for label in labels:
        if label not in label_dict:
            label_dict[label] = map_index
            map_index += 1
        labels[current_index] = label_dict[label]
        current_index += 1
    return labels


# functions to show an image
def imshow(img):
    npimg = img.numpy()  # C*H*W
    print(npimg.transpose((1, 2, 0)).shape, "\n")  # H*W*C
    # plt.imshow 的输入是 H*W*C
    plt.imshow(npimg.transpose((1, 2, 0)), cmap='gray')
    plt.show()


def test():
    trans = transforms.Compose([
        transforms.ToPILImage('L'),  # 转变成PIL image（灰度图）, pixels: 28*28, shape: H*W*C
        transforms.Resize((224, 224), Image.ANTIALIAS),  # pixels: 224*224,  range: [0, 225], shape: H*W*C
        transforms.ToTensor(),  # 转变成Tensor, range: [0, 1.0], shape: C*H*W
        # transforms.Normalize(0.5, 0.5)])      # 单通道用不了这个函数
        transforms.Lambda(lambda x: x.sub(0.5).div(0.5))  # 归一化, range: [-1, 1]
    ])

    trainset = HCL(images_path="./HCL2000-100/HCL2000_100_train.npz",
                   labels_path="./HCL2000-100/HCL2000_100_train_label.npz",
                   transform=trans)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)
    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    npimages = images.numpy()
    print(images.shape)

    # show images
    # make_grid 只接受 4D mini-batch Tensor of shape (B x C x H x W)的输入，但make_grid 会把通道变成3, 而且range会变成255
    imshow(torchvision.utils.make_grid(images, padding=5, pad_value=255))  # shape: C*H*(W*B) 加了一些padding
    # print labels
    print('labels: ', ' '.join('%5s' % labels[j].item() for j in range(4)))


if __name__ == '__main__':
    test()
