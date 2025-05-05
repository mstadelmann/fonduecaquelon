import torch
from monai.losses import DiceLoss
from torchmetrics.image import StructuralSimilarityIndexMeasure
from ui_functions import wprint


class PSNR(torch.nn.Module):
    """
    Pytorch module to calculate PSNR

    https://github.com/fpschill/ac3t/blob/master/ac3t/metrics.py

    :param img: image1 tensor (ground truth)
    :param img2: image2 tensor (predicted)
    :param mode: "norm": for datarange [-1, 1]
                 "HU": for datarange [-1000, 1000]
                 "PSNR_manual": set range manually
    :return: scalar output PSNR(img, img2)

    max is the maximum fluctuation in the input image.
    -> double-precision float: max = 1
    -> 8-bit unsigned int: max = 255

    according to https://pytorch.org/ignite/generated/ignite.metrics.PSNR.html
    The data range of the target image (distance between minimum and maximum possible values).

    i.e. for HU, we consider images in [-1000, 1000], so max = 2000
    -> we could add an offset of 1000 to the images, which would result
    images [0, 2000] without influencing the MSE.
    """

    def __init__(self, mode="norm"):
        super(PSNR, self).__init__()
        self.mode = mode
        self.warning_printed = False  # print warning messages only once
        if mode not in ["PSNR_manual", "norm", "HU"]:
            raise ValueError(
                f"Invalid mode for PSNR, must be 'PSNR_manual', 'norm' or 'HU', but was {mode}"
            )

    def forward(self, img, img2, top_man=None):
        if self.mode == "HU":
            top = 2 * 1000
        elif self.mode == "PSNR_manual":
            if top_man is None:
                raise ValueError(
                    "'top_man' must be manually defined when using PSNR_manual!"
                )
            top = float(top_man)
        else:
            # self.mode == "norm"
            top = 2 * 1

        mse = torch.mean((img - img2) ** 2)
        return 10 * torch.log10(top**2 / mse)


def getLoss(loss_name, device="cuda", largs=None):
    largsd = largs.to_dict()
    if loss_name == "crossentropy":
        loss = torch.nn.CrossEntropyLoss(reduction=largsd.get("reduction", "mean"))
    elif loss_name == "BCEWithLogitsLoss":
        # This loss combines a Sigmoid layer and the BCELoss (Binary Cross Entropy) in one single class.
        loss = torch.nn.BCEWithLogitsLoss()
    elif loss_name == "MSE":
        # return nn.MSELoss(reduction='none')
        loss = torch.nn.MSELoss()
    elif loss_name == "NLL":
        loss = torch.nn.NLLLoss()
    elif loss_name == "MAE":
        # return nn.L1Loss(reduction='none')
        loss = torch.nn.L1Loss()
    elif loss_name == "DiceLoss":
        loss = DiceLoss(to_onehot_y=True, softmax=True)
    elif loss_name == "PSNR_manual":
        loss = PSNR(mode="manual")
    elif loss_name == "PSNR_norm":
        loss = PSNR(mode="norm")
    elif loss_name == "PSNR_hu":
        loss = PSNR(mode="HU")
    elif loss_name == "SSIM_hu":
        loss = StructuralSimilarityIndexMeasure(data_range=2000.0).to(device)
    else:
        raise ValueError(f"Loss {loss_name} is not defined in lossFunctions.py!")

    return loss


def createLoss(experiment):
    for loss_name, largs in experiment.exp_def.losses:
        experiment.losses[loss_name] = getLoss(
            loss_name=largs.name, device=experiment.device, largs=largs
        )
