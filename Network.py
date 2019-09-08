import torchvision
from torchvision.models.detection import FasterRCNN,MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision import transforms
import torch
import os


class MaskNet():
    def __init__(self):
        backbone = torchvision.models.mobilenet_v2(pretrained=True).features
        backbone.out_channels = 1280
        anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256),),
                                           aspect_ratios=((0.5, 1.0, 2.0),))

        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
                                                        output_size=7,
                                                        sampling_ratio=2)
        mask_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],output_size=14,sampling_ratio=2)

        self.net = MaskRCNN(backbone,5,rpn_anchor_generator=anchor_generator,box_roi_pool=roi_pooler,mask_roi_pool=mask_pooler)
        for p in self.net.backbone.parameters():
            p.requires_grad = False

        params = [p for p in self.net.parameters() if p.requires_grad]
        self.optim = torch.optim.SGD(params,lr=0.001,momentum=0.9,weight_decay=0.0005)
        self.lr_schuduler = torch.optim.lr_scheduler.StepLR(self.optim,step_size=3,gamma=0.1)

    def cuda(self):
        self.net = self.net.cuda()

    def train(self,images,targets):
        loss_dict = self.net(images,targets)
        losses = sum(loss for loss in loss_dict.values())
        print(losses.item())
        self.optim.zero_grad()
        losses.backward()
        self.optim.step()

    def save(self):
        torch.save(self.net,os.getcwd()+'/mymodel.pth')

    def load(self):
        self.net = torch.load(os.getcwd()+'/mymodel.pth')




