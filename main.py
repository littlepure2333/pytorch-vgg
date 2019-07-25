import nni
import torch
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from torch import nn, optim

import custom_dataset as ds
import display_grayscale as dg
import vgg

epochs = 5  # number of epochs to train
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
log_interval = 5  # how many batches to wait before logging training status
PATH = "vgg_model.tar"
batch_size = 256
best_accuracy = 0


def train(log_interval, model, device, train_loader, test_loader, optimizer, criterion, epoch):
    model.to(device)
    for batch_idx, (data, target) in enumerate(train_loader):
        model.train()
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # print train progress every log_interval
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

        # record train loss for every 10 batches
        record_index = (epoch - 1) * len(train_loader) + batch_idx + 1
        if record_index % 10 == 0:
            writer.add_scalar('Train/loss', loss.item(), record_index)

        # validate and record validate loss every 10 batches
        if record_index % 10 == 0:
            accuracy = validate(model, device, test_loader, criterion, epoch, record_index)
            nni.report_intermediate_result(accuracy)

    # save the model every epoch
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        # ...
    }, PATH)


def validate(model, device, test_loader, criterion, epoch, record_index):
    # model.to(device)
    model.eval()
    test_loss = 0
    correct = 0
    global best_accuracy
    with torch.no_grad():
        # data_iter = iter(test_loader)
        # data, target = data_iter.next()
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            # validate 10 batches is enough and fast
            if batch_idx == 9:
                break

    length = 10*batch_size
    test_loss /= length
    accuracy = 100. * correct / length
    if accuracy > best_accuracy:
        best_accuracy = accuracy

    print('\nValidate set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, length, accuracy))

    # record accuracy
    writer.add_scalar('validate/accuracy', accuracy, record_index)
    return accuracy


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
    """load NNI for automated machine learning experiments"""
    params = nni.get_next_parameter()
    # or load default parameters
    # params = {
    #     "dropout_rate": 0.5,
    #     "FC_size": 4096,
    #     "learning_rate": 0.01,
    # }

    "record training process"
    global writer
    writer = SummaryWriter(comment='-dr='+str(params['dropout_rate'])+
                                   '-fc='+str(params['FC_size'])+
                                   '-bs='+str(params['batch_size'])+
                                   '-lr='+str(params['learning_rate'])+
                                   '-mo='+str(params['momentum']))

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
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    test_set = ds.HCL(images_path="./HCL2000-100/HCL2000_100_test.npz",
                      labels_path="./HCL2000-100/HCL2000_100_test_label.npz",
                      transform=trans, target_transform=target_trans)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

    "define the network"
    net = vgg.VGG('VGG16', 100, params['dropout_rate'], params['FC_size'])
    # allow parallel compute on multiple GPUs
    net = nn.DataParallel(net)

    "define loss function and optimizer"
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=params['learning_rate'], momentum=params['momentum'])

    "train and validate"
    for epoch in range(1, epochs + 1):
        train(log_interval, net, device, train_loader, test_loader, optimizer, criterion, epoch)
        # validate(net, device, test_loader, criterion, epoch)
    nni.report_final_result(best_accuracy)

    "test"
    # test(net, PATH, device, test_set, 1501)


if __name__ == '__main__':
    main()
