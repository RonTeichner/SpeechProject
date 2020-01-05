import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel

# F.max_pool2d needs kernel_size and stride. If only one argument is passed, 
# then kernel_size = stride

from .audio import MelspectrogramStretch
from torchparse import parse_cfg

# Architecture inspiration from: https://github.com/keunwoochoi/music-auto_tagging-keras
class AudioCRNN(BaseModel):
    def __init__(self, config={}, nDrawsFromSingleEncoderOutput=1, state_dict=None):
        super(AudioCRNN, self).__init__(config)
        
        in_chan = 2 if config['transforms']['args']['channels'] == 'stereo' else 1

        #self.classes = classes
        self.nDrawsFromSingleEncoderOutput = nDrawsFromSingleEncoderOutput
        self.decoderInnerWidth = 100
        self.genderMultivariate = 2
        self.speakerMultivariate = 60
        self.wordMultivariate = 10
        self.nContinuesParameters = 1  # pitch


        self.lstm_units = 64
        self.lstm_layers = 2
        self.spec = MelspectrogramStretch(hop_length=None, 
                                num_mels=128, 
                                fft_length=2048, 
                                norm='whiten', 
                                stretch_param=[0.4, 0.4]).to('cuda')

        # shape -> (channel, freq, token_time)
        self.net = parse_cfg(config['cfg'], in_shape=[in_chan, self.spec.num_mels, 400])
        self.zDim = int(self.net['dense'].linear_0.out_features / 2)
        self.fc6 = nn.Linear(self.zDim, self.genderMultivariate + self.speakerMultivariate + self.wordMultivariate + 2*self.nContinuesParameters)

        # general:
        self.logSoftMax = nn.LogSoftmax(dim=1)
        self.LeakyReLU = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()


    def _many_to_one(self, t, lengths):
        return t[torch.arange(t.size(0)), lengths.long() - 1]

    def modify_lengths(self, lengths):
        def safe_param(elem):
            return elem if isinstance(elem, int) else elem[0]
        
        for name, layer in self.net['convs'].named_children():
            #if name.startswith(('conv2d','maxpool2d')):
            if isinstance(layer, (nn.Conv2d, nn.MaxPool2d)):
                p, k, s = map(safe_param, [layer.padding, layer.kernel_size,layer.stride]) 
                lengths = (lengths + 2*p - k)//s + 1

        #return torch.where(lengths > 0, lengths, torch.tensor(1, device=lengths.device))
        return torch.where(lengths > 0, lengths, (torch.tensor(1, device=lengths.device)).float())

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        #h3 = self.LeakyReLU(self.fc3(z))
        #h4 = self.LeakyReLU(self.fc4(h3))
        #h5 = self.fc5(h4)

        h5 = self.LeakyReLU(self.fc6(z))
        return h5 # h3 here performed good without p(z) constrain

    def encoder(self, batch):
        # x-> (batch, time, channel)
        x, lengths = batch  # unpacking seqs, lengths

        # x-> (batch, channel, time)
        # plt.plot(x[0].cpu().numpy())
        xt = x.float().transpose(1, 2)
        # xt -> (batch, channel, freq, time)

        # import matplotlib.pyplot as plt
        # plt.plot(xt[0, 0].cpu().numpy())
        xt, lengths = self.spec(xt, lengths)
        # plt.imshow(xt[0,0].cpu().numpy())
        # (batch, channel, freq, time)
        xt = self.net['convs'](xt)
        lengths = self.modify_lengths(lengths)

        # xt -> (batch, time, freq, channel)
        x = xt.transpose(1, -1)

        # xt -> (batch, time, channel*freq)
        batch, time = x.size()[:2]
        x = x.reshape(batch, time, -1)
        x_pack = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)

        # x -> (batch, time, lstm_out)
        x_pack, hidden = self.net['recur'](x_pack)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x_pack, batch_first=True)

        # (batch, lstm_out)
        x = self._many_to_one(x, lengths)
        # (batch, classes)
        x = self.net['dense'](x)
        mu, logvar = x[:, :int(x.shape[1]/2)], x[:, -int(x.shape[1]/2):]
        # x = F.log_softmax(x, dim=1)

        return mu, logvar

    def forward(self, batch):
        mu, logvar = self.encoder(batch)
        # plt.plot(batch[0][0].cpu().numpy())
        z = self.reparameterize(mu.repeat(self.nDrawsFromSingleEncoderOutput, 1), logvar.repeat(self.nDrawsFromSingleEncoderOutput, 1))
        decoderOut = self.decode(z)
        genderProbs, speakerProbs, wordProbs, pitchMean, pitchLogVar = decoderOut[:, :self.genderMultivariate], decoderOut[:, self.genderMultivariate:self.genderMultivariate+self.speakerMultivariate], decoderOut[:, self.genderMultivariate+self.speakerMultivariate : self.genderMultivariate+self.speakerMultivariate+self.wordMultivariate], decoderOut[:, self.genderMultivariate+self.speakerMultivariate+self.wordMultivariate : self.genderMultivariate+self.speakerMultivariate+self.wordMultivariate+1], decoderOut[:, self.genderMultivariate+self.speakerMultivariate+self.wordMultivariate+1 : self.genderMultivariate+self.speakerMultivariate+self.wordMultivariate+2]
        return genderProbs, speakerProbs, wordProbs, pitchMean, pitchLogVar, mu, logvar, z


    def predict(self, x):
        with torch.no_grad():
            out_raw = self.forward( x )
            out = torch.exp(out_raw)
            max_ind = out.argmax().item()        
            return self.classes[max_ind], out[:,max_ind].item()


class AudioCNN(AudioCRNN):

    def forward(self, batch):
        x, _, _ = batch
        # x-> (batch, channel, time)
        x = x.float().transpose(1,2)
        # x -> (batch, channel, freq, time)
        x = self.spec(x)                

        # (batch, channel, freq, time)
        x = self.net['convs'](x)

        # x -> (batch, time*freq*channel)
        x = x.view(x.size(0), -1)
        # (batch, classes)
        x = self.net['dense'](x)

        x = F.log_softmax(x, dim=1)

        return x


class AudioRNN(AudioCRNN):

    def forward(self, batch):    
        # x-> (batch, time, channel)
        x, lengths, _ = batch # unpacking seqs, lengths and srs

        # x-> (batch, channel, time)
        x = x.float().transpose(1,2)
        # x -> (batch, channel, freq, time)
        x, lengths = self.spec(x, lengths)                

        # x -> (batch, time, freq, channel)
        x = x.transpose(1, -1)

        # x -> (batch, time, channel*freq)
        batch, time = x.size()[:2]
        x = x.reshape(batch, time, -1)
        x_pack = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
    
        # x -> (batch, time, lstm_out)
        x_pack, hidden = self.net['recur'](x_pack)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x_pack, batch_first=True)
        
        # (batch, lstm_out)
        x = self._many_to_one(x, lengths)
        # (batch, classes)
        x = self.net['dense'](x)

        x = F.log_softmax(x, dim=1)

        return x