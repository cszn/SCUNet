from typing import Optional
import tempfile
import numpy as np
import torch
from cog import BasePredictor, Path, Input, BaseModel, File

from models.network_scunet import SCUNet as net
from utils import utils_image as util
from utils import utils_model


class Output(BaseModel):
    image_with_added_noise: Optional[Path]
    denoised_image: Optional[Path]


class Predictor(BasePredictor):
    def setup(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_paths = {
            "real image denoising": "model_zoo/scunet_color_real_psnr.pth",
            "grayscale images-15": "model_zoo/scunet_gray_15.pth",
            "grayscale images-25": "model_zoo/scunet_gray_25.pth",
            "grayscale images-50": "model_zoo/scunet_gray_50.pth",
            "color images-15": "model_zoo/scunet_color_15.pth",
            "color images-25": "model_zoo/scunet_color_25.pth",
            "color images-50": "model_zoo/scunet_color_50.pth",
        }
        self.models = {}

        for model_name in self.model_paths.keys():
            n_channels = 1 if model_name.startswith("grayscale") else 3
            model = net(in_nc=n_channels, config=[4, 4, 4, 4, 4, 4, 4], dim=64)
            model.load_state_dict(torch.load(self.model_paths[model_name]), strict=True)
            self.models[model_name] = model

    def predict(
        self,
        image: Path = Input(
            description="Input image.",
        ),
        model_name: str = Input(
            choices=[
                "real image denoising",
                "grayscale images-15",
                "grayscale images-25",
                "grayscale images-50",
                "color images-15",
                "color images-25",
                "color images-50",
            ],
            default="real image denoising",
            description="Choose a model. 15, 25 and 50 in grayscale images and color images correspond to the added "
                        "noise level, and the output will show image_with_added_noise and denoised_image.",
        ),
    ) -> Output:

        model = self.models[model_name]
        model.eval()
        for k, v in model.named_parameters():
            v.requires_grad = False
        model = model.to(self.device)

        number_parameters = sum(map(lambda x: x.numel(), model.parameters()))
        print(f"Model params number: {number_parameters}")

        n_channels = 1 if model_name.startswith("grayscale") else 3
        img_L = util.imread_uint(str(image), n_channels=n_channels)

        image_with_added_noise_path = Path(tempfile.mkdtemp()) / "output_noise.png"
        denoised_image_path = Path(tempfile.mkdtemp()) / "output.png"

        if model_name == "real image denoising":
            img_L = util.uint2tensor4(img_L)
            img_L = img_L.to(self.device)
            img_E = model(img_L)
            img_E = util.tensor2uint(img_E)
            util.imsave(img_E, str(denoised_image_path))
            return Output(denoised_image=denoised_image_path)

        img_L = util.uint2single(img_L)
        noise_level = float(model_name.split("-")[-1])
        # degradation process
        np.random.seed(seed=0)  # for reproducibility
        img_L += np.random.normal(0, noise_level / 255.0, img_L.shape)

        img_with_noise = util.single2uint(img_L)

        img_L = util.single2tensor4(img_L)
        img_L = img_L.to(self.device)

        x8 = False
        if not x8 and img_L.size(2) // 8 == 0 and img_L.size(3) // 8 == 0:
            img_E = model(img_L)
        elif not x8 and (img_L.size(2) // 8 != 0 or img_L.size(3) // 8 != 0):
            img_E = utils_model.test_mode(model, img_L, refield=64, mode=5)
        elif x8:
            img_E = utils_model.test_mode(model, img_L, mode=3)

        img_E = util.tensor2uint(img_E)
        util.imsave(img_with_noise, str(image_with_added_noise_path))
        util.imsave(img_E, str(denoised_image_path))

        return Output(image_with_added_noise=image_with_added_noise_path, denoised_image=denoised_image_path)
