import torch
import torch.nn as nn
import timm

try:
    from scripts.config import Config
except:
    from config import Config


# =========================================================
# CONVNEXT-TINY CLASSIFIER
# =========================================================

class ConvNeXtTiny(nn.Module):

    def __init__(self):

        super().__init__()

        # ---------------------------------------------
        # PRETRAINED CONVNEXT
        # ---------------------------------------------

        self.model = timm.create_model(
            Config.MODEL_NAME,
            pretrained=Config.PRETRAINED,
            num_classes=0
        )

        # ---------------------------------------------
        # FEATURE DIMENSION
        # ---------------------------------------------

        in_features = self.model.num_features

        # ---------------------------------------------
        # CLASSIFIER HEAD
        # ---------------------------------------------

        self.classifier = nn.Sequential(

            nn.Dropout(Config.DROPOUT),

            nn.Linear(
                in_features,
                Config.NUM_CLASSES
            )

        )

    # =====================================================
    # FORWARD
    # =====================================================

    def forward(self, x):

        features = self.model(x)

        output = self.classifier(features)

        return output


# =========================================================
# RESIDUAL BLOCK
# =========================================================

class ResidualBlock(nn.Module):

    def __init__(self, channels):

        super().__init__()

        self.block = nn.Sequential(

            nn.ReflectionPad2d(1),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3
            ),

            nn.InstanceNorm2d(channels),

            nn.ReLU(inplace=True),

            nn.Dropout(
                Config.GENERATOR_DROPOUT
            ),

            nn.ReflectionPad2d(1),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3
            ),

            nn.InstanceNorm2d(channels)

        )

    # =====================================================
    # FORWARD
    # =====================================================

    def forward(self, x):

        return x + self.block(x)


# =========================================================
# RESNET GENERATOR
# =========================================================

class ResNetGenerator(nn.Module):

    def __init__(
        self,
        input_channels=3,
        output_channels=3,
        num_residual_blocks=9
    ):

        super().__init__()

        layers = []

        # ---------------------------------------------
        # INITIAL LAYER
        # ---------------------------------------------

        layers += [

            nn.ReflectionPad2d(3),

            nn.Conv2d(
                input_channels,
                64,
                kernel_size=7
            ),

            nn.InstanceNorm2d(64),

            nn.ReLU(inplace=True)

        ]

        # ---------------------------------------------
        # DOWNSAMPLING
        # ---------------------------------------------

        in_channels = 64

        out_channels = in_channels * 2

        for _ in range(2):

            layers += [

                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1
                ),

                nn.InstanceNorm2d(out_channels),

                nn.ReLU(inplace=True)

            ]

            in_channels = out_channels

            out_channels = in_channels * 2

        # ---------------------------------------------
        # RESIDUAL BLOCKS
        # ---------------------------------------------

        for _ in range(num_residual_blocks):

            layers += [

                ResidualBlock(in_channels)

            ]

        # ---------------------------------------------
        # UPSAMPLING
        # ---------------------------------------------

        out_channels = in_channels // 2

        for _ in range(2):

            layers += [

                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1
                ),

                nn.InstanceNorm2d(out_channels),

                nn.ReLU(inplace=True)

            ]

            in_channels = out_channels

            out_channels = in_channels // 2

        # ---------------------------------------------
        # OUTPUT LAYER
        # ---------------------------------------------

        layers += [

            nn.ReflectionPad2d(3),

            nn.Conv2d(
                in_channels,
                output_channels,
                kernel_size=7
            ),

            nn.Tanh()

        ]

        self.model = nn.Sequential(*layers)

    # =====================================================
    # FORWARD
    # =====================================================

    def forward(self, x):

        return self.model(x)


# =========================================================
# PATCHGAN DISCRIMINATOR
# =========================================================

class PatchGANDiscriminator(nn.Module):

    def __init__(
        self,
        input_channels=3
    ):

        super().__init__()

        def discriminator_block(
            in_filters,
            out_filters,
            normalize=True
        ):

            layers = [

                nn.Conv2d(
                    in_filters,
                    out_filters,
                    kernel_size=4,
                    stride=2,
                    padding=1
                )

            ]

            if normalize:

                layers.append(
                    nn.InstanceNorm2d(out_filters)
                )

            layers.append(
                nn.LeakyReLU(
                    0.2,
                    inplace=True
                )
            )

            return layers

        self.model = nn.Sequential(

            *discriminator_block(
                input_channels,
                64,
                normalize=False
            ),

            *discriminator_block(
                64,
                128
            ),

            *discriminator_block(
                128,
                256
            ),

            *discriminator_block(
                256,
                512
            ),

            nn.Conv2d(
                512,
                1,
                kernel_size=4,
                padding=1
            )

        )

    # =====================================================
    # FORWARD
    # =====================================================

    def forward(self, x):

        return self.model(x)


# =========================================================
# MAIN TEST
# =========================================================

if __name__ == "__main__":

    device = Config.DEVICE

    # ---------------------------------------------
    # CONVNEXT TEST
    # ---------------------------------------------

    classifier = ConvNeXtTiny().to(device)

    x = torch.randn(
        2,
        3,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE
    ).to(device)

    y = classifier(x)

    print("\nConvNeXt Output Shape:")
    print(y.shape)

    # ---------------------------------------------
    # GENERATOR TEST
    # ---------------------------------------------

    generator = ResNetGenerator().to(device)

    fake = generator(x)

    print("\nGenerator Output Shape:")
    print(fake.shape)

    # ---------------------------------------------
    # DISCRIMINATOR TEST
    # ---------------------------------------------

    discriminator = PatchGANDiscriminator().to(device)

    pred = discriminator(fake)

    print("\nDiscriminator Output Shape:")
    print(pred.shape)

    print("\nModels initialized successfully.\n")