import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(inplace=True),
            nn.Linear(in_features, in_features),
            nn.BatchNorm1d(in_features)
        )

    def forward(self, x):
        return x + self.block(x)

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, feature_dim, dims):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential()
        for i in range(len(dims) + 1):
            if i == 0:
                self.encoder.add_module('Linear%d' % i, nn.Linear(input_dim, dims[i]))
                self.encoder.add_module('relu%d' % i, nn.ReLU())
                self.encoder.add_module('batchnorm%d' % i, nn.BatchNorm1d(dims[i]))  # 添加 BatchNorm 层
                self.encoder.add_module('dropout%d' % i, nn.Dropout(0.5))  # 增加 Dropout 比例
            elif i == len(dims):
                self.encoder.add_module('Linear%d' % i, nn.Linear(dims[i - 1], feature_dim))
                self.encoder.add_module('relu%d' % i, nn.ReLU())
            else:
                self.encoder.add_module('Linear%d' % i, nn.Linear(dims[i - 1], dims[i]))
                self.encoder.add_module('relu%d' % i, nn.ReLU())
                self.encoder.add_module('batchnorm%d' % i, nn.BatchNorm1d(dims[i]))  # 添加 BatchNorm 层
                self.encoder.add_module('dropout%d' % i, nn.Dropout(0.5))  # 增加 Dropout 比例
                self.encoder.add_module('residual%d' % i, ResidualBlock(dims[i]))  # 添加残差块

    def forward(self, x):
        return self.encoder(x)

class AutoDecoder(nn.Module):
    def __init__(self, input_dim, feature_dim, dims):
        super(AutoDecoder, self).__init__()
        self.decoder = nn.Sequential()
        dims = list(reversed(dims))
        for i in range(len(dims) + 1):
            if i == 0:
                self.decoder.add_module('Linear%d' % i, nn.Linear(feature_dim, dims[i]))
                self.decoder.add_module('relu%d' % i, nn.ReLU())
                self.decoder.add_module('batchnorm%d' % i, nn.BatchNorm1d(dims[i]))  # 添加 BatchNorm 层
                self.decoder.add_module('dropout%d' % i, nn.Dropout(0.5))  # 增加 Dropout 比例
            elif i == len(dims):
                self.decoder.add_module('Linear%d' % i, nn.Linear(dims[i - 1], input_dim))
                self.decoder.add_module('relu%d' % i, nn.ReLU())
            else:
                self.decoder.add_module('Linear%d' % i, nn.Linear(dims[i - 1], dims[i]))
                self.decoder.add_module('relu%d' % i, nn.ReLU())
                self.decoder.add_module('batchnorm%d' % i, nn.BatchNorm1d(dims[i]))  # 添加 BatchNorm 层
                self.decoder.add_module('dropout%d' % i, nn.Dropout(0.5))  # 增加 Dropout 比例
                self.decoder.add_module('residual%d' % i, ResidualBlock(dims[i]))  # 添加残差块

    def forward(self, x):
        return self.decoder(x)

class CLJENetwork(nn.Module):
    def __init__(self, num_views, input_sizes, dims, dim_high_feature, dim_low_feature, num_clusters):
        super(CLJENetwork, self).__init__()
        self.encoders = list()
        self.decoders = list()
        for idx in range(num_views):
            self.encoders.append(AutoEncoder(input_sizes[idx], dim_high_feature, dims))
            self.decoders.append(AutoDecoder(input_sizes[idx], dim_high_feature, dims))
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)

        self.label_learning_module = nn.Sequential(
            nn.Linear(dim_high_feature, dim_low_feature),
            nn.ReLU(),
            nn.BatchNorm1d(dim_low_feature),
            nn.Dropout(p=0.5),
            nn.Linear(dim_low_feature, num_clusters),
            nn.Softmax(dim=1)
        )

    def forward(self, data_views):
        lbps = list()
        dvs = list()
        features = list()

        num_views = len(data_views)
        for idx in range(num_views):
            data_view = data_views[idx]
            high_features = self.encoders[idx](data_view)
            label_probs = self.label_learning_module(high_features)
            data_view_recon = self.decoders[idx](high_features)
            features.append(high_features)
            lbps.append(label_probs)
            dvs.append(data_view_recon)

        return lbps, dvs, features
