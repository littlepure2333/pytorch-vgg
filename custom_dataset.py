import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset


class HCL(Dataset):
    def __init__(self, images_path, labels_path, transforms=None, target_transforms=None):
        images_data = np.load(images_path)
        labels_data = np.load(labels_path)
        images = images_data["arr_0"]
        images = images.reshape((images.shape[0], 28, 28))
        self.images = torch.from_numpy(images)
        labels = labels_data["arr_0"]
        self.labels = torch.from_numpy(labels.astype(np.int16))
        self.transforms = transforms
        self.target_transforms = transforms

    def __getitem__(self, index):
        image = self.images[index]
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


# functions to show an image
def imshow(img):
    npimg = img.numpy()
    print(npimg.transpose((1, 2, 0)).shape, "\n")
    plt.imshow(npimg.transpose((1, 2, 0)), cmap='gray')
    plt.show()


def test():
    trainset = HCL(images_path="./HCL2000-100/HCL2000_100_train.npz",
                   labels_path="./HCL2000-100/HCL2000_100_train_label.npz")

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)
    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    print(images.shape)
    print(images.unsqueeze(1).shape)
    print(torchvision.utils.make_grid(images.unsqueeze(1)).shape)

    # show images
    # make_grid 只接受 4D mini-batch Tensor of shape (B x C x H x W)的输入，unsqueeze增加C这一维度
    imshow(torchvision.utils.make_grid(images.unsqueeze(1), pad_value=255))
    # print labels
    print('labels: ', ' '.join('%5s' % labels[j].item() for j in range(4)))


if __name__ == '__main__':
    test()
