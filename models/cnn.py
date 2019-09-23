import torch
from torchvision.models.resnet import resnet18
from models.inception import beheaded_inception_v3
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class ConvNet(torch.nn.Module):
    def __init__(self, rnn_hidden=256, out_size=12):
        super(ConvNet, self).__init__()
        self.rnn_hidden = rnn_hidden
        self.num_layers = 4
        self.out_size = out_size
        self.lstm_input_size = out_size
        self.encoder = resnet18(pretrained=True)
        # Disable grad
        # for param in self.encoder.parameters():
        #     param.requires_grad = False
        # for param in self.encoder.layer4.parameters():
        #     param.requires_grad = True
        self.encoder.fc = torch.nn.Linear(self.encoder.fc.in_features, 512)
        self.cnn2h0 = torch.nn.Sequential(torch.nn.Dropout(0.4),
                                          torch.nn.Linear(512, self.num_layers * self.rnn_hidden))
        self.cnn2c0 = torch.nn.Sequential(torch.nn.Dropout(0.4),
                                          torch.nn.Linear(512,
                                                          self.num_layers * self.rnn_hidden))
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
        h0 = self.cnn2h0(features).view(self.num_layers, img.shape[0], self.rnn_hidden)
        c0 = self.cnn2c0(features).view(self.num_layers, img.shape[0], self.rnn_hidden)

        nums = pack_padded_sequence(nums, lengths, batch_first=True, enforce_sorted=False)
        out, self.hidden_w = self.lstm(nums, (h0, c0))
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=8)
        out = self.linear(out)
        return out

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.num_layers, batch_size, self.rnn_hidden)
        return hidden

    def predict(self, imgs):
        features = self.encoder(imgs)
        sampled_ids = []
        h0 = self.cnn2h0(features).view(self.num_layers, imgs.shape[0], self.rnn_hidden)
        c0 = self.cnn2c0(features).view(self.num_layers, imgs.shape[0], self.rnn_hidden)
        states = (h0, c0)
        inputs = torch.zeros(imgs.shape[0], 1, self.out_size).cuda()
        inputs[:, 0, 0] = 1
        for i in range(8):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            _, predicted = outputs.max(1)
            sampled_ids.append(predicted)
            inputs = outputs.unsqueeze(1)
            print(inputs)

        sampled_ids = torch.stack(sampled_ids, 1)
        return sampled_ids.detach().cpu().numpy()
