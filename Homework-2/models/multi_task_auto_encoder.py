import torch.nn as nn


class MultiTaskAutoEncoder(nn.Module):
    def __init__(
        self,
        latent_size,
        input_channels=3,
        encoder_channels=[16, 32],
        kernel_size=3,
        stride=2,
        padding=1,
        num_classes=3,
        flattened_latent_size=2048,
    ):
        super(MultiTaskAutoEncoder, self).__init__()

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

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(flattened_latent_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        latent_flat = latent.view(latent.size(0), -1)
        classification = self.classifier(latent_flat)

        return reconstructed, classification
