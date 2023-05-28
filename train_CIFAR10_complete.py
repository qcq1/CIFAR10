import os.path


import torchvision.datasets
from torch.utils.tensorboard import SummaryWriter

from QCQ_VGG16 import *
from QCQ_ResNet18 import *
from torch.utils.data import DataLoader

# 训练数据集
train_data = torchvision.datasets.CIFAR10("CIFAR10/CIFAR10_train", train=True,
                                          transform=torchvision.transforms.ToTensor(), download=True)
# 测试数据集
test_data = torchvision.datasets.CIFAR10("CIFAR10/CIFAR10_test", train=False,
                                         transform=torchvision.transforms.ToTensor(), download=True)

# 数据集长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

# dataloader加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 定义运行设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 创建神经网络模型
vgg = [96, 96, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

select=input("选择模型:\n"
             "1.VGG16\n"
             "2.RestNet18\n")
if select=='1':
    qcq = QCQ(vgg)
    model_name='VGG16'
    learning_rate = 1e-2
    optimizer = torch.optim.SGD(qcq.parameters(), lr=learning_rate,weight_decay=5e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.4, last_epoch=-1)
    # 训练轮数
    epoch = 10
else:
    qcq = ResNet18()
    model_name = 'RestNet18'
    # 优化器,学习率
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(qcq.parameters(), lr=learning_rate)
    # 训练轮数
    epoch = 50
qcq = qcq.to(device)

# 创建损失函数
loss_fun = nn.CrossEntropyLoss().to(device)
loss_fun = loss_fun.to(device)



# 设置训练网络的一些参数
# 训练次数
total_train_step = 0
# 测试次数
total_test_step = 0


# 添加tensorboard
if not os.path.exists('logs'):
    os.mkdir('logs')
writer = SummaryWriter('logs')

for i in range(epoch):
    print("--------第%5d   轮训练开始--------" % (i + 1))

    # 训练步骤开始
    qcq.train()
    for data in train_dataloader:
        imgs, targets = data

        imgs = imgs.to(device)
        targets = targets.to(device)

        outputs = qcq(imgs)
        loss = loss_fun(outputs, targets)
        # 模型优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step += 1
        if total_train_step % 100 == 0:
            print('训练次数：%-8d,Loss：%f' % (total_train_step, loss.item()))
            writer.add_scalar('train_loss', loss.item(), total_train_step)

    # 测试步骤开始
    qcq.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data

            imgs = imgs.to(device)
            targets = targets.to(device)

            outputs = qcq(imgs)
            loss = loss_fun(outputs, targets)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy
    print('整体测试集上test_loss：%f' % total_test_loss)
    print('整体测试集上正确率：%f' % (total_accuracy / test_data_size))
    writer.add_scalar('test_loss', total_test_loss, total_test_step)
    writer.add_scalar('test_accuracy_rate', total_accuracy / test_data_size, total_test_step)
    if select=='1': scheduler.step()
    total_test_step += 1

    # 模型每轮保存
    if not os.path.exists('model'):
        os.mkdir('model')
    torch.save(qcq.state_dict(), 'model/'+model_name+'_%d_epoch.pth' % (i+1))
    print("模型自动保存成功")

writer.close()
