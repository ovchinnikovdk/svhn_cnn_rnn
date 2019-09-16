import torch
from torchvision.models.resnet import resnet18
from torch.nn.utils.rnn import pack_padded_sequence


class ConvNet(torch.nn.Module):
    def __init__(self, rnn_hidden=256, out_size=12):
        super(ConvNet, self).__init__()
        self.rnn_hidden = rnn_hidden
        self.num_layers = 16
        self.out_size = out_size
        self.lstm_input_size = 64
        self.encoder = resnet18(pretrained=True)
        # for param in self.encoder.parameters():
        #     param.requires_grad = False
        print(self.encoder.fc.in_features, self.num_layers * self.rnn_hidden)
        self.encoder.fc = torch.nn.Sequential(
            torch.nn.Linear(self.encoder.fc.in_features, self.encoder.fc.in_features),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(self.encoder.fc.in_features, self.lstm_input_size - self.out_size))
        self.lstm = torch.nn.LSTM(input_size=self.lstm_input_size,
                                  hidden_size=rnn_hidden,
                                  num_layers=self.num_layers,
                                  dropout=0.4,
                                  batch_first=True)
        self.linear = torch.nn.Sequential(torch.nn.Linear(rnn_hidden, rnn_hidden),
                                          torch.nn.Dropout(0.4),
                                          torch.nn.Linear(rnn_hidden, out_size))
        self.hidden_w = None

    def forward(self, data):
        img, nums = data[0], data[1]
        features = self.encoder(img)
        start = torch.zeros(nums.shape[0], 1, nums.shape[2]).cuda()
        start[0, 0, 0] = 1
        nums = torch.cat((start, nums), 1)
        a = torch.zeros(nums.shape[0], nums.shape[1], self.lstm_input_size).cuda()
        a[:, :, :nums.shape[2]] = nums
        for i in range(nums.shape[0]):
            a[i, :, nums.shape[2]:] = features[i]
        nums = a
        h = torch.zeros(self.num_layers, nums.shape[0], self.rnn_hidden).cuda()
        out, self.hidden_w = self.lstm(nums, (h, torch.zeros_like(h)))
        out = self.linear(out)[:, :-1, :]
        return torch.nn.Softmax(dim=2)(out)

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.num_layers, batch_size, self.rnn_hidden)
        return hidden

    def predict(self, imgs):
        features = self.encoder(imgs)
        sampled_ids = []
        states = (torch.zeros(self.num_layers, imgs.shape[0], self.rnn_hidden).cuda(),
                  torch.zeros(self.num_layers, imgs.shape[0], self.rnn_hidden).cuda())
        inputs = torch.zeros(imgs.shape[0], 1, self.lstm_input_size).cuda()
        inputs[0, 0, 0] = 1
        inputs[:, 0, self.out_size:] = features
        for i in range(7):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            print(outputs)
            _, predicted = outputs.max(1)
            sampled_ids.append(predicted)
            inputs = torch.cat((outputs, features), 1).unsqueeze(1)
        sampled_ids = torch.stack(sampled_ids, 1)
        return sampled_ids.detach().cpu().numpy()
