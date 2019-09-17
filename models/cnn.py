import torch
from torchvision.models.resnet import resnet50
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class ConvNet(torch.nn.Module):
    def __init__(self, rnn_hidden=256, out_size=12):
        super(ConvNet, self).__init__()
        self.rnn_hidden = rnn_hidden
        self.num_layers = 8
        self.out_size = out_size
        self.lstm_input_size = out_size
        self.encoder = resnet50(pretrained=True)
        for param in list(self.encoder.parameters())[:-2]:
            param.requires_grad = False
        print(self.encoder.fc.in_features, self.num_layers * self.rnn_hidden)
        self.encoder.fc = torch.nn.Sequential(
            torch.nn.Linear(self.encoder.fc.in_features, self.encoder.fc.in_features),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(self.encoder.fc.in_features, self.rnn_hidden * self.num_layers))
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
        img, nums, lengths = data[0], data[1], data[2]
        features = self.encoder(img)
        start = torch.zeros(nums.shape[0], 1, nums.shape[2]).cuda()
        start[:, 0, 0] = 1
        nums = nums[:, :-1, :]
        nums = torch.cat((start, nums), 1)
        nums = pack_padded_sequence(nums, lengths, batch_first=True, enforce_sorted=False)
        h = features.permute(1, 0).contiguous().view(self.num_layers, img.shape[0], self.rnn_hidden)
        out, self.hidden_w = self.lstm(nums, (h, torch.zeros_like(h)))
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=8)
        out = self.linear(out)
        return out

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.num_layers, batch_size, self.rnn_hidden)
        return hidden

    def predict(self, imgs):
        features = self.encoder(imgs)
        sampled_ids = []
        h = features.permute(1, 0).contiguous().view(self.num_layers, imgs.shape[0], self.rnn_hidden)
        states = (h,
                  torch.zeros_like(h))
        inputs = torch.zeros(imgs.shape[0], 1, self.out_size).cuda()
        inputs[:, 0, 0] = 1
        for i in range(8):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            # print(outputs)
            _, predicted = outputs.max(1)
            sampled_ids.append(predicted)
            inputs = outputs.unsqueeze(1)
            inputs = torch.zeros_like(inputs)
            # print(inputs.shape, predicted.shape)
            for i in range(inputs.shape[0]):
                inputs[i, 0, predicted[i].item()] = 1
        sampled_ids = torch.stack(sampled_ids, 1)
        return sampled_ids.detach().cpu().numpy()
