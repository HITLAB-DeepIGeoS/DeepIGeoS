import torch
import torch.nn as nn


class P_RNet3D(nn.Module):
    '''
    3D P/R-Net Architectures
    '''
    def __init__(self, c_in, c_blk, n_classes, init_weights=True):
        super().__init__()

        # Conv blocks (resolution preserving)
        self.block1 = nn.Sequential(
            nn.Conv3d(c_in, c_blk, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1), dilation=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(c_blk, c_blk, kernel_size=(3, 3, 1), stride=1, padding=(1, 1, 0), dilation=(1, 1, 1)),
            nn.ReLU(),
        )
        self.block1_downsample = nn.Sequential(
            nn.Conv3d(c_blk, c_blk//4, kernel_size=(1, 1, 1), stride=1, padding=0, dilation=1),
            nn.ReLU(),
        )

        self.block2 = nn.Sequential(
            nn.Conv3d(c_blk, c_blk, kernel_size=(3, 3, 3), stride=1, padding=(2, 2, 2), dilation=(2, 2, 2)),
            nn.ReLU(),
            nn.Conv3d(c_blk, c_blk, kernel_size=(3, 3, 1), stride=1, padding=(2, 2, 0), dilation=(2, 2, 1)),
            nn.ReLU(),
        )
        self.block2_downsample = nn.Sequential(
            nn.Conv3d(c_blk, c_blk//4, kernel_size=(1, 1, 1), stride=1, padding=0, dilation=1),
            nn.ReLU(),
        )

        self.block3 = nn.Sequential(
            nn.Conv3d(c_blk, c_blk, kernel_size=(3, 3, 3), stride=1, padding=(4, 4, 4), dilation=(4, 4, 4)),
            nn.ReLU(),
            nn.Conv3d(c_blk, c_blk, kernel_size=(3, 3, 1), stride=1, padding=(4, 4, 0), dilation=(4, 4, 1)),
            nn.ReLU(),
            nn.Conv3d(c_blk, c_blk, kernel_size=(3, 3, 1), stride=1, padding=(4, 4, 0), dilation=(4, 4, 1)),
            nn.ReLU(),
        )
        self.block3_downsample = nn.Sequential(
            nn.Conv3d(c_blk, c_blk//4, kernel_size=(1, 1, 1), stride=1, padding=0, dilation=1),
            nn.ReLU(),
        )

        self.block4 = nn.Sequential(
            nn.Conv3d(c_blk, c_blk, kernel_size=(3, 3, 3), stride=1, padding=(8, 8, 8), dilation=(8, 8, 8)),
            nn.ReLU(),
            nn.Conv3d(c_blk, c_blk, kernel_size=(3, 3, 1), stride=1, padding=(8, 8, 0), dilation=(8, 8, 1)),
            nn.ReLU(),
            nn.Conv3d(c_blk, c_blk, kernel_size=(3, 3, 1), stride=1, padding=(8, 8, 0), dilation=(8, 8, 1)),
            nn.ReLU(),
        )
        self.block4_downsample = nn.Sequential(
            nn.Conv3d(c_blk, c_blk//4, kernel_size=(1, 1, 1), stride=1, padding=0, dilation=1),
            nn.ReLU(),
        )

        self.block5 = nn.Sequential(
            nn.Conv3d(c_blk, c_blk, kernel_size=(3, 3, 3), stride=1, padding=(16, 16, 16), dilation=(16, 16, 16)),
            nn.ReLU(),
            nn.Conv3d(c_blk, c_blk, kernel_size=(3, 3, 1), stride=1, padding=(16, 16, 0), dilation=(16, 16, 1)),
            nn.ReLU(),
            nn.Conv3d(c_blk, c_blk, kernel_size=(3, 3, 1), stride=1, padding=(16, 16, 0), dilation=(16, 16, 1)),
            nn.ReLU(),
        )
        self.block5_downsample = nn.Sequential(
            nn.Conv3d(c_blk, c_blk//4, kernel_size=(1, 1, 1), stride=1, padding=0, dilation=1),
            nn.ReLU(),
        )

        # Classifier block (resolution preserving)
        self.block6 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv3d((c_blk//4)*5, c_blk, kernel_size=(1, 1, 1), stride=1, padding=0, dilation=1),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Conv3d(c_blk, n_classes, kernel_size=(3, 3, 3), stride=1, padding=1, dilation=1),
        )

        if init_weights:
            self.initialize_weights()

    def forward(self, inputs):
        out_blk1 = self.block1(inputs)
        out_blk2 = self.block2(out_blk1)
        out_blk3 = self.block3(out_blk2)
        out_blk4 = self.block4(out_blk3)
        out_blk5 = self.block5(out_blk4)

        out_blks = torch.cat([self.block1_downsample(out_blk1),
                              self.block2_downsample(out_blk2),
                              self.block3_downsample(out_blk3),
                              self.block4_downsample(out_blk4),
                              self.block5_downsample(out_blk5)], dim=1)
                              
        out_logits = self.block6(out_blks)

        return out_logits

    def initialize_weights(self):
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)


def build_model(config):
    model = P_RNet3D(config.model.c_in, 
                     config.model.c_blk,
                     config.model.n_classes,
                     config.model.init_weights)
                     
    if config.exp.multi_gpu:
        pass
    else:
        model = model.to(config.exp.device)

    return model
    