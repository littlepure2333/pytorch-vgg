import nni
import torch
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from torch import nn, optim
import custom_dataset as ds
import display_grayscale as dg
import vgg
import train

batch_size = 256
PATH = train.PATH
device = torch.device("cpu")


def test(model, model_path, device, test_set, index):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    with torch.no_grad():
        img, target = test_set.__getitem__(index)
        # display the image
        npimg = torch.squeeze(img).numpy()
        dg.display(npimg)
        # compute the output
        img, target = img.to(device), target.to(device)
        output = model(img.unsqueeze(0))
        pred = output.argmax(dim=1, keepdim=True)
        print("Predicted:", pred.item(), "\t GroundTruth:", target.item())


def main():
    """load default parameters"""
    params = train.params

    "load and normalize HCL2000-100 dataset"
    trans = transforms.Compose([
        # transforms.ToPILImage('L'),  # 转变成PIL image（灰度图）, pixels: 28*28, shape: H*W*C
        # transforms.Resize((224, 224), Image.ANTIALIAS),  # pixels: 224*224,  range: [0, 225], shape: H*W*C
        transforms.ToTensor(),  # 转变成Tensor, range: [0, 1.0], shape: C*H*W
        transforms.Lambda(lambda x: x.sub(0.5).div(0.5))])  # 归一化, range: [-1, 1]

    target_trans = transforms.Lambda(lambda x: x.long())

    test_set = ds.HCL(images_path="./HCL2000-100/HCL2000_100_test.npz",
                      labels_path="./HCL2000-100/HCL2000_100_test_label.npz",
                      transform=trans, target_transform=target_trans)

    "define the network"
    net = vgg.VGG('VGG16', 100, params['dropout_rate'], params['FC_size'])

    "test one of the 30000 images"
    test(net, PATH, device, test_set, 20000)


if __name__ == '__main__':
    main()
