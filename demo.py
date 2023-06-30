import os
from tkinter import scrolledtext

import PIL
import torchvision.transforms
from PIL import ImageTk, ImageFont, ImageDraw
from PIL import Image as Img

import tkinter.filedialog
from tkinter import *
from QCQ_VGG16 import *
from QCQ_ResNet18 import *

CIFAR10_class = ['airplane', 'automobile', 'brid', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
vgg = [96, 96, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
model_path = ''
image_path = ''
qcq_test = None


def readImage(img_path='img/test.png'):
    img = Img.open(img_path).convert('RGB')
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.Resize((32, 32)), torchvision.transforms.ToTensor()])
    img = transform(img)
    print(img.shape)
    img = torch.reshape(img, (1, 3, 32, 32))
    return img


def DrawImageTxt(imageFile, targetImageFile, txtnum):
    # 设置字体大小
    font = PIL.ImageFont.truetype('img/abc.ttf', 50)
    # 打开文件
    im = Img.open(imageFile)
    # 字体坐标
    draw = PIL.ImageDraw.Draw(im)
    draw.text((0, 0), txtnum, (255, 0, 0), font=font)
    # 保存
    im.save(targetImageFile)
    # 关闭
    im.close()


def chosemodelfun():
    global model_path
    model_path = tkinter.filedialog.askopenfilename(title="选择一个.pth的模型文件", filetypes=[('PTH', '*.pth')])
    outtext.insert(END, "选择模型：" + model_path + '\n' + '\n' + '\n')


def choseimgfun():
    global image_path
    image_path = tkinter.filedialog.askopenfilename(title="选择一个.png的测试图片", filetypes=[('PNG', '*.png')])
    img_open = Img.open(image_path)
    img = ImageTk.PhotoImage(img_open.resize((200, 200)))
    img_bef.config(image=img)
    img_bef.image = img
    outtext.insert(END, "选择图片：" + image_path + '\n' + '\n' + '\n')


def showimgafter(imgaft_path):
    img_open = Img.open(imgaft_path)
    img = ImageTk.PhotoImage(img_open.resize((200, 200)))
    img_aft.config(image=img)
    img_aft.image = img


def test():
    global qcq_test
    device = 'cuda'
    if str(model_path).__contains__("VGG16"):
        qcq_test = QCQ(vgg)
    elif str(model_path).__contains__("RestNet18"):
        qcq_test = ResNet18()
    qcq_test.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    img = readImage(image_path)
    qcq_test.eval()
    with torch.no_grad():
        output = qcq_test(img)
        pre = output.argmax(1)
        txtnum = CIFAR10_class[pre.item()]
        image_split = image_path.split('/')
        image_name = image_split[image_split.__len__() - 1].split('.')[0]
        targetImageFile = os.path.abspath(".") + '/img/' + image_name + '_pre.png'
        DrawImageTxt(image_path, targetImageFile, txtnum)
        print(output)
        print(pre)
        print(txtnum)
        outtext.insert(END, "识别结果：" + str(output) + '\n' + str(pre) + '\n' + str(txtnum) + '\n' + '\n' + '\n')
    showimgafter(targetImageFile)


if __name__ == '__main__':
    # ui设计
    root = Tk()
    root.title('选择预训练的模型检测图片')
    root.geometry('828x512')

    lb0 = Label(root)
    lb0.config(text="通过训练好的模型识别十种事物：飞机,汽车,鸟,猫,鹿,狗,青蛙,马,船,卡车", font=32)
    lb0.place(relx=0.1, rely=0, relwidth=0.8)

    chosemodel = Button(root, text="选择本地模型文件", command=chosemodelfun)
    chosemodel.place(relx=0.1, rely=0.1, relwidth=0.3)

    choseimg = Button(root, text="选择本地测试图片", command=choseimgfun)
    choseimg.place(relx=0.6, rely=0.1, relwidth=0.3)

    begin = Button(root, text="开始识别", command=test)
    begin.place(relx=0.35, rely=0.2, relwidth=0.3)

    img_bef = Label(root)
    img_bef.place(relx=0.05, rely=0.3, relwidth=0.4)
    img_aft = Label(root)
    img_aft.place(relx=0.5, rely=0.3, relwidth=0.4)

    outtext = scrolledtext.ScrolledText(root)
    outtext.place(relx=0.1, rely=0.7, relwidth=0.8, relheight=0.29)
    root.mainloop()
