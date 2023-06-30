import os.path
import threading
from tkinter import scrolledtext

import torchvision.datasets
from torch.utils.tensorboard import SummaryWriter

from QCQ_VGG16 import *
from QCQ_ResNet18 import *
from torch.utils.data import DataLoader

from tkinter import *

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

epoch = None
qcq = None
optimizer = None
scheduler = None
model_name = None


def train(select):
    global epoch, qcq, optimizer, scheduler, model_name
    if select == 1:
        qcq = QCQ(vgg)
        model_name = 'VGG16'
        learning_rate = 1e-2
        optimizer = torch.optim.SGD(qcq.parameters(), lr=learning_rate, weight_decay=5e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.4, last_epoch=-1)
        # 训练轮数
        epoch = 10
    elif select == 2:
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
        ui_log.insert(END, "--------第%5d   轮训练开始--------\n" % (i + 1))

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
                ui_log.insert(END, '训练次数：%-8d,Loss：%f\n' % (total_train_step, loss.item()))
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
        ui_log.insert(END, '整体测试集上test_loss：%f\n' % total_test_loss)
        print('整体测试集上正确率：%f' % (total_accuracy / test_data_size))
        ui_log.insert(END, '整体测试集上正确率：%f\n' % (total_accuracy / test_data_size))
        writer.add_scalar('test_loss', total_test_loss, total_test_step)
        writer.add_scalar('test_accuracy_rate', total_accuracy / test_data_size, total_test_step)
        if select == '1':
            scheduler.step()
        total_test_step += 1

        # 模型每轮保存
        if not os.path.exists('model'):
            os.mkdir('model')
        torch.save(qcq.state_dict(), 'model/' + model_name + '_%d_epoch.pth' % (i + 1))
        print("模型自动保存成功")
        ui_log.insert(END, "模型自动保存成功\n")
    writer.close()
    print("模型训练完成")
    ui_log.insert(END, "模型训练完成\n")


# ui设计
def run(*args):
    btn1.config(state=DISABLED)
    btn2.config(state=DISABLED)
    t = threading.Thread(target=train, args=args)  # 开线程防止UI卡死
    t.setDaemon(True)
    t.start()


root = Tk()
root.title('选择模型并训练')
root.geometry('828x512')

lb0 = Label(root)
lb0.config(text="-选择一个模型进行训练-", font=32, anchor=CENTER)
lb0.place(relx=0.35, rely=0, relwidth=0.3)

ui_log = scrolledtext.ScrolledText(root)
ui_log.place(relx=0.1, rely=0.3, relwidth=0.8)

btn1 = Button(root, text='VGG16', command=lambda: run(1))
btn1.place(relx=0.1, rely=0.1, relwidth=0.3, relheight=0.1)
btn2 = Button(root, text='RestNet18', command=lambda: run(2))
btn2.place(relx=0.6, rely=0.1, relwidth=0.3, relheight=0.1)

root.mainloop()
