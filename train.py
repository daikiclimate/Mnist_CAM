import argparse
import time
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import torch.optim as optim

from net import Net, Net_FC


def parser():
    '''
    argument
    '''
    parser = argparse.ArgumentParser(description='PyTorch MNIST')
    parser.add_argument('--epochs', '-e', type=int, default=2,
                        help='number of epochs to train (default: 2)')
    parser.add_argument('--lr', '-l', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    args = parser.parse_args()
    return args


def main():
    '''
    main
    '''
    args = parser()

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, ), (0.5, ))])


    trainset = MNIST(root='./data',
                     train=True,
                     download=True,
                     transform=transform)
    testset = MNIST(root='./data',
                    train=False,
                    download=True,
                    transform=transform)
                    

    trainloader = DataLoader(trainset,
                             batch_size=100,
                             shuffle=True,
                             num_workers=2)
    testloader = DataLoader(testset,
                            batch_size=100,
                            shuffle=False,
                            num_workers=2)

    classes = tuple(np.linspace(0, 9, 10, dtype=np.uint8))

    # model
    net = Net_FC()

    # define loss function and optimier
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(),
                          lr=args.lr, momentum=0.99, nesterov=True)
    optimizer = optim.Adam(net.parameters(), lr = 0.001)

    # train
    for epoch in range(args.epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader, 0):
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            print('\r[{:d}, {:5d}] loss: {:.4f}'.format(epoch+1, i+1, running_loss/(i+1)),end="")
        print("")
    print('Finished Training')


    # test
    correct = 0
    total = 0
    with torch.no_grad():
        for (images, labels) in testloader:
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy: {:.2f} %'.format(100 * float(correct/total)))

    # print(net.return_CAM().shape)
    # tmp = net.return_CAM()
    # tmp = tmp[0][0]
    # tmp = tmp - np.min(tmp)
    # tmp = tmp / np.max(tmp)
    # tmp = tmp * 255
    # tmpimg = Image.fromarray(tmp).convert("L")
    # tmpimg.save("a.jpg")
    saveCAM(np.array(images), labels.numpy(), net.return_CAM())

from PIL import Image
def saveCAM(imgs,labels,cams):
    
    inv_transform = transforms.Normalize(

        mean=(-0.5/0.5,),
        std=(1/0.5,)
    )
    print(labels.shape)
    print(np.array(cams).shape)
    for i in range(10):
        index = labels == i
        img = imgs[index][0]
        img = inv_transform(torch.tensor(img)).numpy()
        img = img.reshape(28,28)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # 色をRGBに変換
        cv2.imwrite("result/"+str(i)+"src_img.png",img*255)

        cam = cams[index]
        for j in range(10):
            c = cv2.resize(cam[i][j], (28,28))
            jetcam = cv2.applyColorMap(np.uint8(255 * c), cv2.COLORMAP_JET)  # モノクロ画像に疑似的に色をつける
            # jetcam = cv2.cvtColor(jetcam, cv2.COLOR_BGR2RGB)  # 色をRGBに変換
            jetcam = (np.float32(jetcam)/2 + img*122 )   # もとの画像に合成

            cv2.imwrite("result/"+str(i)+str(j)+"cam.png",jetcam)



if __name__ == '__main__':
    start_time = time.time()
    main()
    print('time: {:.1f} [sec]'.format(time.time() - start_time))
