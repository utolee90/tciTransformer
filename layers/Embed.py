import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import numpy as np

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(
                x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            # the potential to take covariates (e.g. timestamps) as tokens
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1)) 
        # x: [Batch Variate d_model]
        return self.dropout(x)


# TCN based inverted
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.1, bias=True):
        super(TemporalBlock, self).__init__()
        
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias))
        # print("conv1 shape",self.conv1.weight.shape)
        self.chomp1 = Chomp1d(padding)

        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias))
        self.chomp2 = Chomp1d(padding)
        # print("conv2 shape",self.conv2.weight.shape)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1, self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1, bias=bias) if n_inputs != n_outputs else None
        # print("downsample shape",self.downsample.weight.shape)
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
        # print("net shape",self.downsample.weight.shape)

    def forward(self, x):
        # print(x.size(), self.conv1.in_channels)
        if x.size()[1] != self.conv1.in_channels:
            x = x.permute(0, 2, 1) # 추가
        # print("Input size:", x.size())
        out = self.net(x)
        # print("Output size after conv layers:", out.size())
        res = x if self.downsample is None else self.downsample(x)
        # print("Output size after downsample:", res.size())

        # Pad or truncate res to match the size of out along the third dimension
        if res.size(2) != out.size(2):
            if res.size(2) < out.size(2):
                res = torch.nn.functional.pad(res, (0, out.size(2) - res.size(2)))
            else:
                res = res[:, :, :out.size(2)]

        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.1, bias=True):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout, bias=bias)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def arithmetic_terms(a, b, r):
    """
    a와 b 사이에 r개의 텀을 가진 등차수열을 생성하는 함수.
    반환되는 수들은 자연수여야 합니다.
    
    Parameters:
    a (int): 시작 값
    b (int): 끝 값
    r (int): 등차수열의 텀 개수
    
    Returns:
    list: 등차수열을 이루는 r개의 자연수 리스트
    """
    # linspace를 사용하여 a와 b 사이의 r개 점을 생성
    sequence = np.linspace(a, b, r)
    
    # 각 값을 반올림하여 자연수로 변환
    sequence = np.round(sequence).astype(int)
    
    # 중복 제거 및 순서 유지
    unique_sequence = list(dict.fromkeys(sequence))
    
    return unique_sequence


class DataEmbedding_inverted_TCN(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1, num_layers=3, kernel_size=2, tcn_dropout_rate=0.1, uniform_layer= True, tcn_bias=True):
        super(DataEmbedding_inverted_TCN, self).__init__()
        num_channels = list(np.array(arithmetic_terms(c_in/4, d_model/4, num_layers+1)[1:])*4) # 첫 번째는 제외
        if uniform_layer:
            self.value_embedding = TemporalConvNet(c_in, [d_model]*num_layers, kernel_size, tcn_dropout_rate, tcn_bias)
        else:
            self.value_embedding = TemporalConvNet(c_in, num_channels, kernel_size, tcn_dropout_rate, tcn_bias)  # Example: 3 layers of TCN with d_model channels
            
        # self.value_embedding = TemporalConvNet(c_in, [d_model]*3, 3, 0.005) # kernerl size 3, dropout 0.01
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        # print("x",x.shape)
        # x = x.permute(0, 2, 1)  # x: [Batch, Variate, Time]
        # print("permuted x",x.shape)
        # if x_mart is not None:
        #    print("x_mark", x_mark.shape)
        # if x_mark is not None:
        #     x = torch.cat([x, x_mark.permute(0, 2, 1)], 1)  # Concatenate along the variate dimension
        # print(self.dropout(x).shape) # check
        # return self.dropout(x)
        if x_mark is not None:
            x = torch.cat([x, x_mark], 2)
        x = self.value_embedding(x)
        
        # print(self.dropout(x.permute(0,2,1)).shape)
        return self.dropout(x.permute(0,2,1))


# TCN 원조 코드 사용
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        # chomp_size(padding size): (kernel_size-1) * dilation_size 
        self.chomp_size = chomp_size  

    def forward(self, x):
        # output: (N, C_in, L_in - chomp_size)
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock1(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2, bias=True):
        super(TemporalBlock1, self).__init__()
        #--------------------- Dilated Causal Convolution --------------------- 
        '''
        input: (N, C_in, L_in) -> (N, C_out, L_in)
        output sequence의 길이는 변하지 않는다. 
        '''
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation, bias=bias))
        self.chomp1 = Chomp1d(padding)
        #----------------------------------------------------------------------
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        #--------------------- Dilated Causal Convolution --------------------- 
        '''
        input: (N, C_in, L_in) -> output: (N, C_out, L_in)
        output sequence의 길이는 변하지 않는다. 
        '''
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation, bias=bias))
        self.chomp2 = Chomp1d(padding)
        #----------------------------------------------------------------------
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        # Dilated Causal Conv -> WeightedNorm -> ReLU -> Dropout -> ... (논문의 Residual block 구조와 동일)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        
        # Residual Connections 
        '''  
        만약 Dilated Causal Conv를 적용하기 전의 input channel과 적용한 후의 
        output channel이 달라질 경우를 대비하여 추가적인 1*1 convolution을 추가해줌.

        input: (N, C_in, L_in) -> output: (N, C_out, L_in)
        output sequence의 길이는 변하지 않는다. (1*1 convolution)
        '''  
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1, bias=bias) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet1(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet1, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            # dilation factor는 layer의 깊이에 지수적으로 증가 (2^i, i: layer depth) 
            dilation_size = 2 ** i 
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            # 설정된 layer의 깊이만큼 TemporalBlock 생성 
            layers += [TemporalBlock1(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet1(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, inputs):
        """
        Inputs have to have dimension (N, C_in, L_in)
        Flatten된 MNIST의 경우에는 (batch_size, 1, 784)
        """
        # input 디멘션 바꾸기. (N, )
        y1 = self.tcn(inputs.permute(0,2,1))  # input should have dimension (N, C, L)
        o = self.linear(y1.permute(0,2,1)) # sequence의 마지막 time step로 linear계산 -> 분류예측
        return F.log_softmax(o, dim=1)