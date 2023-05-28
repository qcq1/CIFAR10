import os

import PIL
import torchvision.transforms
from PIL import Image


from QCQ_VGG16 import *
from QCQ_ResNet18 import *

CIFAR10_class = ['airplane','automobile','brid','cat','deer','dog','frog','horse','ship','truck']
vgg = [96, 96, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

def readImage(img_path='img/test.png'):
    img = Image.open(img_path).convert('RGB')
    transform= torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),torchvision.transforms.ToTensor()])
    img = transform(img)
    print(img.shape)
    img = torch.reshape(img, (1, 3, 32, 32))
    return img


def DrawImageTxt(imageFile, targetImageFile, txtnum):
    # 设置字体大小
    font = PIL.ImageFont.truetype('img/abc.ttf', 50)
    # 打开文件
    im = Image.open(imageFile)
    # 字体坐标
    draw = PIL.ImageDraw.Draw(im)
    draw.text((0, 0), txtnum, (255, 255, 0), font=font)
    # 保存
    im.save(targetImageFile)
    # 关闭
    im.close()


if __name__=='__main__':
    print("通过训练好的模型识别十种事物：飞机,汽车,鸟,猫,鹿,狗,青蛙,马,船,卡车")
    device = 'cuda'
    models=[]
    i=1
    for root, dirs, files in os.walk('model'):
        for file in files:
            if file.__contains__('.pth'):
                file_path=root+'/'+file
                models.append(file_path)
                print(f'{i}.'+file)
                i+=1
    if models.__len__()==0:
        print('model文件夹中没有pth模型文件')
    else:
        select = int(input('选择一个模型\n'))
        model_path=models[select-1]
        if model_path.__contains__('VGG16'):
            qcq_test=QCQ(vgg)
        elif model_path.__contains__('RestNet18'):
            qcq_test=ResNet18()
        else:
            print('选择的模型名称中既不包含"VGG16"，也不包含"RestNet18"')

        qcq_test.load_state_dict(torch.load(model_path, map_location=torch.device(device)))

        imgs=[]
        i=1
        for root, dirs, files in os.walk('img'):
            for file in files:
                if file.__contains__('.png') or file.__contains__('.jpg'):
                    file_path = root + '/' + file
                    imgs.append(file_path)
                    print(f'{i}.' + file)
                    i+=1
        if imgs.__len__() == 0:
            print('img文件夹中没有图片')
        else:
            select = int(input('选择一个测试图片,\n'))
            image_path = imgs[select-1]
            image_name,image_type = image_path.split('.')
            targetImageFile=image_name+'_pre.png'
            img=readImage(image_path)
            qcq_test.eval()
            with torch.no_grad():
                output = qcq_test(img)
                pre=output.argmax(1)
                txtnum = CIFAR10_class[pre.item()]
                DrawImageTxt(image_path, targetImageFile, txtnum)
                print(output)
                print(pre)
                print(txtnum)
            Image.open(targetImageFile).show()







