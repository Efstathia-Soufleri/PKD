'''
Reimplemented from reading paper titled
dos Santos, Samuel Felipe, Nicu Sebe, and Jurandy Almeida. 
"CV-C3D: action recognition on compressed videos with convolutional 3d networks." 
In 2019 32nd SIBGRAPI Conference on Graphics, Patterns and Images (SIBGRAPI), 
pp. 24-30. IEEE, 2019.
'''

from c3d.c3d_base import C3D
import torch.nn as nn

class CVC3D(nn.Module):
    def __init__(self, num_segments, num_classes):
        super(CVC3D, self).__init__()

        self.i_model = C3D(in_channels=num_segments, num_classes=num_classes)
        self.r_model = C3D(in_channels=num_segments, num_classes=num_classes)
        self.mv_model = C3D(in_channels=num_segments, num_classes=num_classes)

    def forward(self, input_i, input_mv, input_r):
        out1 = self.i_model(input_i)
        out2 = self.mv_model(input_mv)
        out3 = self.r_model(input_r)
        return out1 + out2 + out3
