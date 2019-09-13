from models.cnn import ConvNet
from torchvision.transforms import ToTensor
import torch
import cv2
import os
import numpy as np

model_path = 'logdir/checkpoints/best.pth'
model = ConvNet(rnn_hidden=96)
model.load_state_dict(torch.load(model_path)['model_state_dict'])
model.cuda()
model.eval()

print('Model loaded successfully...')

if not os.path.exists('predicted'):
    os.mkdir('predicted')


def pred2number(predictions):
    print(predictions)
    res = []
    for i in range(np.where(predictions == 11)[0][0]):
        res.append(predictions[i] if predictions[i] != 10 else 0)
    return res


test_path = os.path.join('data', 'test')
imgs = list(filter(lambda x: '.png' in x, os.listdir(test_path)))
for img_name in np.random.choice(imgs, size=20):
    img = cv2.imread(os.path.join(test_path, img_name))
    img = cv2.resize(img, (64, 32))
    img = ToTensor()(img).cuda()
    img = img[None]
    predictions = model.predict(img)
    predictions = pred2number(predictions[0])
    print(img_name, predictions)
    cv2.imwrite('predicted/'+ img_name + '_' + ''.join([str(num) for num in predictions]) + '.png',
                cv2.imread(os.path.join(test_path, img_name)))