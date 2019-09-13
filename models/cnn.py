import torch
from torchvision.models.resnet import resnet34
from torch.nn.utils.rnn import pack_padded_sequence


class ConvNet(torch.nn.Module):
    def __init__(self, rnn_hidden=256, out_size=12):
        super(ConvNet, self).__init__()
        self.rnn_hidden = rnn_hidden
        self.num_layers = 48
        self.out_size = out_size
        self.lstm_input_size = 12
        self.encoder = resnet34(pretrained=True)
        # for param in self.encoder.parameters():
        #     param.requires_grad = False
        print(self.encoder.fc.in_features, self.num_layers * self.rnn_hidden)
        self.encoder.fc = torch.nn.Sequential(
            torch.nn.Linear(self.encoder.fc.in_features, self.encoder.fc.in_features),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(self.encoder.fc.in_features, self.num_layers * self.rnn_hidden))
        self.lstm = torch.nn.LSTM(input_size=self.lstm_input_size,
                                  hidden_size=rnn_hidden,
                                  num_layers=self.num_layers,
                                  dropout=0.4,
                                  batch_first=True)
        self.linear = torch.nn.Sequential(torch.nn.Linear(rnn_hidden, rnn_hidden // 2),
                                          torch.nn.Dropout(0.4),
                                          torch.nn.Linear(rnn_hidden // 2, out_size))
        # self.linear = torch.nn.Linear(rnn_hidden, out_size)
        self.hidden_w = None

    def forward(self, data):
        img, nums = data[0], data[1]
        features = self.encoder(img)
        features = features.permute(1, 0)
        features = features.contiguous().view(self.num_layers, img.shape[0], self.rnn_hidden)
        start = torch.zeros(nums.shape[0], 1, nums.shape[2]).cuda()
        start[0, 0, 0] = 1
        nums = torch.cat((start, nums), 1)
        # nums = nums[:, :-1, :]
        # nums = pack_padded_sequence(nums, lens, batch_first=True, enforce_sorted=False)
        out, self.hidden_w = self.lstm(nums, (features, torch.zeros_like(features)))
        out = self.linear(out)
        return out[:, :-1, :]

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.num_layers, batch_size, self.rnn_hidden)
        return hidden

    def predict(self, imgs):
        features = self.encoder(imgs)
        sampled_ids = []
        states = features.permute(1, 0)\
            .contiguous()\
            .view(self.num_layers, imgs.shape[0], self.rnn_hidden)
        states = (states, torch.zeros_like(states))
        inputs = torch.zeros(imgs.shape[0], 1, self.out_size).cuda()
        inputs[0, 0, 0] = 1
        for i in range(7):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            _, predicted = outputs.max(1)
            sampled_ids.append(predicted)
            inputs = outputs.unsqueeze(1)
        sampled_ids = torch.stack(sampled_ids, 1)
        return sampled_ids.detach().cpu().numpy()
