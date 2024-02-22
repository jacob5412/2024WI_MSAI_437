import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(
        self,
        latent_size,
        input_channels=3,
        encoder_channels=[16, 32],
        kernel_size=3,
        stride=2,
        padding=1,
    ):
        super(AutoEncoder, self).__init__()

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv2d(
                input_channels, encoder_channels[0], kernel_size, stride, padding
            ),
            nn.ReLU(),
            nn.Conv2d(
                encoder_channels[0], encoder_channels[1], kernel_size, stride, padding
            ),
            nn.ReLU(),
            nn.Conv2d(encoder_channels[1], latent_size, kernel_size, stride, padding),
            nn.LeakyReLU(),
        )

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                latent_size,
                encoder_channels[1],
                kernel_size,
                stride,
                padding,
                output_padding=1,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                encoder_channels[1],
                encoder_channels[0],
                kernel_size,
                stride,
                padding,
                output_padding=1,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                encoder_channels[0],
                input_channels,
                kernel_size,
                stride,
                padding,
                output_padding=1,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
