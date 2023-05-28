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
    device = 'cuda'
    model_path = 'model/'+input('输入model下一个模型的文件名，如VGG16_10_epoch\n')+'.pth'
    if model_path.__contains__('VGG16'):
        qcq_test=QCQ(vgg)
    elif model_path.__contains__('RestNet18'):
        qcq_test=ResNet18()

    qcq_test.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    image = 'img/test' #修改这里以更改测试图片
    imageType = '.png'
    imageFile=image+imageType
    targetImageFile = image+'_pre.png'
    img=readImage(imageFile)
    qcq_test.eval()
    with torch.no_grad():
        output = qcq_test(img)
        pre=output.argmax(1)
        txtnum = CIFAR10_class[pre.item()]
        DrawImageTxt(imageFile, targetImageFile, txtnum)
        print(output)
        print(pre)
        print(txtnum)








