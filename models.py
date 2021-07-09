import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers=2, dropout=0.5):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.rnn = nn.LSTM(input_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [src len, batch size, input_dim]
        outputs, (hidden, cell) = self.rnn(src)
        #print(outputs.shape, hidden.shape, cell.shape)
        # outputs = [src len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]
        # outputs are always from the top hidden layer
        return hidden

class Discriminator(nn.Module):
  def __init__(self, channels_img, audio_dim, hidden_dim, z_dim, features_d):
    super(Discriminator, self).__init__()
    self.enc = Encoder(audio_dim, hidden_dim)
    self.classifier = nn.Sequential(
        nn.Linear((z_dim+2*hidden_dim), 1),
        nn.Sigmoid()
    )
    self.disc = nn.Sequential(
        #input: N x channels_image x 64 x 64
        nn.Conv3d(channels_img, features_d, kernel_size = (1,4,4), stride = (1,2,2), padding = (0,1,1)), #32x32
        nn.LeakyReLU(0.2),
        self._block(features_d, features_d*2, (4,4,4), (2,2,2), (1,1,1)),   #16x16
        self._block(features_d*2, features_d*4, (4,4,4), (2,2,2), (1,1,1)),   #8x8
        self._block(features_d*4, features_d*8, (4,4,4), (2,2,2), (1,1,1)),   #4x4
        nn.Conv3d(features_d*8, z_dim, kernel_size = (4,4,4), stride = (2,2,2), padding = (0,0,0))
        #nn.Sigmoid(),
    )

  def _block(self, in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                         nn.BatchNorm3d(out_channels),
                         nn.LeakyReLU(0.2),
                         )
  def forward(self, x, audio):
    y = self.enc(audio)
    y = y.permute(1, 0, 2)
    y = y.reshape(y.shape[0], y.shape[1] * y.shape[2])
    x = self.disc(x)
    x = x.reshape(x.shape[0], x.shape[1])
    q = torch.cat((x, y), 1)
    return self.classifier(q)

class Generator(nn.Module):
  def __init__(self, channels_img, audio_dim, hidden_dim, z_dim, features_g):
    super(Generator, self).__init__()
    self.enc = Encoder(audio_dim, hidden_dim)
    self.gen = nn.Sequential(
        #Input: N x z_dim x 1 x 1
        self._block((z_dim+2*hidden_dim), features_g*16, kernel_size=(4,4,4), stride=(1,1,1), padding=(0,0,0)),  # N x f_g*16 x 4 x 4
        self._block(features_g*16, features_g*8,kernel_size=(4,4,4), stride=(2,2,2), padding=(1,1,1)),  # 8 x 8
        self._block(features_g*8, features_g*4, kernel_size=(4,4,4), stride=(2,2,2), padding=(1,1,1)),  #16 x 16
        self._block(features_g*4, features_g*2, kernel_size=(4,4,4), stride=(2,2,2), padding=(1,1,1)),   # 32 X 32
        nn.ConvTranspose3d(features_g*2, channels_img, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1) ),  # 64 x 64
        nn.Tanh(),   # [-1, 1]
    )
  def _block(self, in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                         nn.BatchNorm3d(out_channels),
                         nn.ReLU(),
                         )
  def forward(self, x, audio):
    y = self.enc(audio)
    y = y.permute(1, 0, 2)
    y = y.reshape(y.shape[0], y.shape[1] * y.shape[2])

    q = torch.cat((x, y), 1)
    q = q.reshape(q.shape[0], q.shape[1],1,1,1)
    return self.gen(q)

# fixed_noise = torch.randn(16, 256)
# gen = Generator(1, 513, 128, 256, 64)
# audio = torch.rand(63, 16, 513)
# q = torch.rand(16, 256)
# t = gen(q, audio)
# print(t.shape)

# dis = Discriminator(1, 513, 128, 256, 64)
# audio = torch.rand(63, 32, 513)
# img = torch.rand(32, 1, 32, 64, 64)
# t = dis(img, audio)
# print(t.shape)