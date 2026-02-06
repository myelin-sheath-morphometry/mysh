import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, p_dropout=0):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=p_dropout),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=p_dropout)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(pl.LightningModule):
    def __init__(self,
                 n_channels=3,
                 n_classes=1,
                 bilinear=True,
                 architecture=(32, 64, 128, 256, 256),
                 l1_lambda=0,
                 l2_weight_decay=0,
                 p_dropout=0,
                 use_boundary_aware_loss=False,
                 fp_weight=2.0,
                 learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.l1_lambda = l1_lambda
        self.l2_weight_decay = l2_weight_decay
        self.p_dropout = p_dropout
        self.fp_weight = fp_weight
        self.architecture = architecture
        self.learning_rate = learning_rate

        self.inc = DoubleConv(n_channels, architecture[0], p_dropout=self.p_dropout)

        self.down_path = nn.ModuleList()
        for i in range(len(architecture) - 1):
            self.down_path.append(
                nn.Sequential(
                    nn.MaxPool2d(2),
                    DoubleConv(architecture[i], architecture[i + 1], p_dropout=self.p_dropout)
                )
            )

        self.up_path = nn.ModuleList()
        for i in range(len(architecture) - 1, 0, -1):
            if bilinear:
                self.up_path.append(
                    nn.ModuleList([
                        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                        DoubleConv(architecture[i] + architecture[i - 1], architecture[i - 1], p_dropout=self.p_dropout)
                    ])
                )
            else:
                self.up_path.append(
                    nn.ModuleList([
                        nn.ConvTranspose2d(architecture[i], architecture[i - 1], kernel_size=2, stride=2),
                        DoubleConv(architecture[i], architecture[i - 1], p_dropout=self.p_dropout)
                    ])
                )

        self.outc = nn.Conv2d(architecture[0], n_classes, kernel_size=1)

        if use_boundary_aware_loss:
            self.dice_loss = BoundaryAwareDiceLoss(fp_weight=self.fp_weight)
        else:
            self.dice_loss = DiceLoss()

    def forward(self, x):
        features = [self.inc(x)]

        for down in self.down_path:
            features.append(down(features[-1]))

        x = features[-1]
        for up, skip in zip(self.up_path, reversed(features[:-1])):
            x = up[0](x)
            diff_y = skip.size()[2] - x.size()[2]
            diff_x = skip.size()[3] - x.size()[3]
            x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2,
                          diff_y // 2, diff_y - diff_y // 2])
            x = torch.cat([skip, x], dim=1)
            x = up[1](x)

        return torch.sigmoid(self.outc(x))

    def _get_l1_regularization(self):
        l1_reg = torch.tensor(0., requires_grad=True)
        for param in self.parameters():
            l1_reg = l1_reg + torch.norm(param, 1)
        return self.l1_lambda * l1_reg

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        dice_loss = self.dice_loss(y_hat, y)
        l1_loss = self._get_l1_regularization()
        loss = dice_loss + l1_loss
        self.log('train_loss', loss)
        self.log('train_dice_loss', dice_loss)
        self.log('train_l1_loss', l1_loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.dice_loss(y_hat, y)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.dice_loss(y_hat, y)
        self.log('test_loss', loss)
        return loss

    def predict_step(self, batch, batch_idx):
        """
        Prediction step that handles both:
        - Inference-only data: batch is a tuple with single tensor (x,)
        - Evaluation data: batch is a tuple (x, y)

        Returns:
            For inference: tensor of predictions
            For evaluation: dict with 'predictions' and 'targets'
        """
        if isinstance(batch, (list, tuple)):
            x = batch[0]
            y = batch[1] if len(batch) > 1 else None
        else:
            x = batch
            y = None

        y_hat = self(x)

        if y is None:
            return y_hat
        return {'predictions': y_hat, 'targets': y}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate,
                                     weight_decay=self.l2_weight_decay)
        return optimizer


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = pred.view(-1)
        target = target.view(-1)
        intersection = (pred * target).sum()

        return 1 - ((2. * intersection + self.smooth) /
                    (pred.sum() + target.sum() + self.smooth))


class BoundaryAwareDiceLoss(nn.Module):
    def __init__(self, smooth=1.0, fp_weight=2.0, gaussian_sigma=1.0):
        super().__init__()
        self.smooth = smooth
        self.fp_weight = fp_weight  # Weight for false positives
        self.gaussian_sigma = gaussian_sigma
        kernel = self._create_gaussian_kernel()
        self.register_buffer("gaussian_kernel", kernel)

    def _create_gaussian_kernel(self, kernel_size=5):
        """Creates a 2D Gaussian kernel for smoothing"""
        x = torch.linspace(-2, 2, kernel_size)
        y = torch.linspace(-2, 2, kernel_size)
        x_grid, y_grid = torch.meshgrid(x, y, indexing='ij')
        kernel = torch.exp(-(x_grid ** 2 + y_grid ** 2) / (2 * self.gaussian_sigma ** 2))
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, kernel_size, kernel_size)

    def _apply_gaussian_smoothing(self, mask):
        """Applies Gaussian smoothing to the binary mask"""
        padded_mask = F.pad(mask, (2, 2, 2, 2), mode='reflect')
        smoothed = F.conv2d(padded_mask, self.gaussian_kernel)
        return smoothed

    def forward(self, pred, target):
        target_smooth = self._apply_gaussian_smoothing(target)

        pred = pred.view(-1)
        target = target.view(-1)
        target_smooth = target_smooth.view(-1)

        boundary_weight = 1 - 4 * target_smooth * (1 - target_smooth)

        intersection = ((pred * target) * boundary_weight).sum()
        false_positives = ((pred * (1 - target)) * boundary_weight).sum()

        numerator = 2 * intersection + self.smooth
        denominator = pred.sum() + target.sum() + self.fp_weight * false_positives + self.smooth

        loss = 1 - numerator / denominator

        return loss.mean()

def batch_inference(data: torch.Tensor, model: UNet, batch_size: int) -> torch.Tensor:
    """
    Perform batch inference using PyTorch model.

    Parameters:
        data (torch.Tensor): Input data tensor
        model (UNet): PyTorch UNet model
        batch_size (int): Batch size for inference

    Returns:
        torch.Tensor: Concatenated model outputs
    """
    if len(data) == 0:
        return torch.tensor([])

    model.eval()
    device = model.device
    outputs = []

    with torch.inference_mode():
        autocast_device = 'cuda' if device.type == 'cuda' else 'cpu'
        with torch.autocast(device_type=autocast_device, dtype=torch.float16, enabled=(device.type == 'cuda')):
            for i in range(0, len(data), batch_size):
                x_batch = data[i:i + batch_size].to(device, non_blocking=True)
                output = model(x_batch)
                outputs.append(output.cpu())

    return torch.cat(outputs, dim=0)
