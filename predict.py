from models.cnn import ConvNet
from torchvision.transforms import ToTensor
from lib.dataset import get_transforms, HouseNumberTrainDataset
import torch
import cv2
import os
import numpy as np

model_path = 'logdir/checkpoints/best.pth'
model = ConvNet(rnn_hidden=32)
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
    # return predictions


test_path = os.path.join('data', 'test')
dataset = HouseNumberTrainDataset(test_path, 'data/test.mat', get_transforms)

for idx in np.random.choice(list(range(len(dataset))), size=30):
    img = dataset[idx][0][0].cuda()
    print(img.min(), img.max())
    predictions = model.predict(img[None])
    print(predictions)
    predictions = pred2number(predictions[0])
    print(idx, predictions)
    res = ''.join([str(num) for num in predictions])
    np_img = img.permute(1, 2, 0).detach().cpu().numpy()
    cv2.imshow(res, np_img)
    cv2.waitKey()
    # cv2.imwrite(f"predicted/{idx}_{res}.png",
    #             np_img)