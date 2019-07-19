import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn, optim

import vgg
import custom_dataset as ds
import torchvision.transforms as transforms

epochs = 5  # number of epochs to train
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
log_interval = 20  # how many batches to wait before logging training status
PATH = "vgg_model.tar"


def train(log_interval, model, device, train_loader, optimizer, criterion, epoch):
    model.to(device)
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    # save the model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        # ...
    }, PATH)


def validate(model, device, test_loader, criterion):
    model.to(device)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        data_iter = iter(test_loader)
        data, target = data_iter.next()
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += criterion(output, target).item()  # sum up batch loss
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= data.size(0)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, data.size(0),
        100. * correct / data.size(0)))


def test(model, device, test_loader, criterion):
    model.to(device)
    model.eval()


def main():
    """only for test"""
    # vgg.test()

    "load and normalize HCL2000-100 dataset"
    trans = transforms.Compose([
        # transforms.ToPILImage('L'),  # 转变成PIL image（灰度图）, pixels: 28*28, shape: H*W*C
        # transforms.Resize((224, 224), Image.ANTIALIAS),  # pixels: 224*224,  range: [0, 225], shape: H*W*C
        transforms.ToTensor(),  # 转变成Tensor, range: [0, 1.0], shape: C*H*W
        transforms.Lambda(lambda x: x.sub(0.5).div(0.5))])  # 归一化, range: [-1, 1]

    target_trans = transforms.Lambda(lambda x: x.long())

    train_set = ds.HCL(images_path="./HCL2000-100/HCL2000_100_train.npz",
                       labels_path="./HCL2000-100/HCL2000_100_train_label.npz",
                       transform=trans, target_transform=target_trans)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=256, shuffle=True)

    test_set = ds.HCL(images_path="./HCL2000-100/HCL2000_100_test.npz",
                      labels_path="./HCL2000-100/HCL2000_100_test_label.npz",
                      transform=trans, target_transform=target_trans)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=True)

    "define the network"
    net = vgg.VGG('VGG16', 100)
    # allow parallel compute on multiple GPUs
    net = nn.DataParallel(net)

    "define loss function and optimizer"
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    "train and validate"
    for epoch in range(1, epochs + 1):
        train(log_interval, net, device, train_loader, optimizer, criterion, epoch)
        validate(net, device, test_loader, criterion)


if __name__ == '__main__':
    main()
