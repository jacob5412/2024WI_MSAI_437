import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(
        self,
        latent_size,
        input_channels=3,
        hidden_layer_1=32,
        hidden_layer_2=16,
        kernel_size=3,
        stride=2,
        padding=1,
    ):
        super(AutoEncoder, self).__init__()

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden_layer_1, kernel_size, stride, padding),
            nn.ReLU(),
            nn.Conv2d(hidden_layer_1, hidden_layer_2, kernel_size, stride, padding),
            nn.ReLU(),
            nn.Conv2d(hidden_layer_2, latent_size, kernel_size, stride, padding),
            nn.LeakyReLU(),
        )

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                latent_size,
                hidden_layer_2,
                kernel_size,
                stride,
                padding,
                output_padding=1,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                hidden_layer_2,
                hidden_layer_1,
                kernel_size,
                stride,
                padding,
                output_padding=1,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                hidden_layer_1,
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
