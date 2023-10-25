import rclpy
from rclpy import qos
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from PIL import Image as PImage
import torch
import torch.nn as nn
import torchvision.transforms as transforms

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


ROBOT = '/qbert'
WEIGHTS_PATH = 'best.pth'
SCALE = (120,160)
IN_CHANNELS = 3
OUT_CHANNELS = 3

TIMER_PERIOD = 0.1



class BallDetector(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Output layer
        self.outconv = nn.ConvTranspose2d(128, out_channels, kernel_size=1)

        # Pre-processing transform
        self.transform = transforms.Compose([
            transforms.Resize(SCALE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def forward(self, x):
        enc = self.encoder(x) # Encoder
        bott = self.bottleneck(enc) # Bottleneck
        dec = self.decoder(bott) # Decoder
        dec = nn.functional.interpolate(dec, scale_factor=2, mode='bilinear', align_corners=False)

        # Skip connection and final prediction
        skip = torch.cat([enc, dec], dim=1)
        out = self.outconv(skip)
        out = nn.functional.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False)

        return out

    def initialize(self, path):
        device = torch.device('cpu')
        state_dict = torch.load(path, map_location=device)
        self.load_state_dict(state_dict)
        self.eval()



class Sweeper(Node):

    def __init__(self):
        super().__init__('sweeper')
        self.bridge = CvBridge()
        self.detector = BallDetector(IN_CHANNELS, OUT_CHANNELS)
        self.yellows = []
        self.blues = []

        self.num = 1

        self.detector.initialize(WEIGHTS_PATH)

        self.camera = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.cameraCallback,
            qos.qos_profile_sensor_data)
        self.timer = self.create_timer(TIMER_PERIOD, self.timerCallback)

    def cameraCallback(self, msg):
        raw = self.bridge.imgmsg_to_cv2(msg, 'rgb8')

    def timerCallback(self):
        raw = cv2.imread(f'ins/ball{self.num}.jpg')
        raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        raw = PImage.fromarray(raw)
        img = self.detector.transform(raw).unsqueeze(0)
        mask = self.detector(img).squeeze(0).detach()
        mask = torch.argmax(torch.softmax(mask, dim=0), dim=0).numpy().astype(np.uint8)
        colors = plt.cm.get_cmap('viridis',3)
        custom_cmap = ListedColormap([colors(i) for i in range(3)])
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(raw.resize((160,120)))
        axes[0].set_title('Image')
        axes[1].imshow(mask, cmap=custom_cmap)
        axes[1].set_title('Mask')
        for ax in axes.ravel(): ax.axis('off')
        plt.savefig(f'outs/result{self.num}.jpg')
        plt.close()
        self.num += 1


def main(args=None):
    rclpy.init(args=args)
    sweeper = Sweeper()
    rclpy.spin(sweeper)
    sweeper.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
