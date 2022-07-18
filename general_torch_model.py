import torch
import numpy as np
import torch.nn as nn
from torchvision.transforms import ToPILImage, ToTensor
try:
    from cStringIO import StringIO as BytesIO
except ImportError:
    from io import BytesIO
from scipy import ndimage
from PIL import Image


class GeneralTorchModel(nn.Module):
    def __init__(self, model, n_class=10, im_mean=None, im_std=None, defense_method=None):
        super(GeneralTorchModel, self).__init__()
        self.model = model
        self.model.eval()
        self.num_queries = 0
        self.im_mean = im_mean
        self.im_std = im_std
        self.n_class = n_class
        self.defense_method = defense_method
        self.guass_sigma = 2

    def forward(self, image):
        if len(image.size()) != 4:
            image = image.unsqueeze(0)
        if self.defense_method is not None:
            if self.defense_method == 'jepg':
                image = self.jpeg_compression(image)
            elif self.defense_method == 'gauss':
                image = self.gaussian_filter(image, self.guass_sigma)
            else:
                raise NameError('False defense method')
        image = self.preprocess(image)
        logits = self.model(image)
        return logits

    def preprocess(self, image):
        if isinstance(image, np.ndarray):
            processed = torch.from_numpy(image).type(torch.FloatTensor)
        else:
            processed = image

        if self.im_mean is not None and self.im_std is not None:
            im_mean = torch.tensor(self.im_mean).cuda().view(1, processed.shape[1], 1, 1).repeat(
                processed.shape[0], 1, 1, 1)
            im_std = torch.tensor(self.im_std).cuda().view(1, processed.shape[1], 1, 1).repeat(
                processed.shape[0], 1, 1, 1)
            processed = (processed - im_mean) / im_std
        return processed

    def predict_prob(self, image):

        if len(image.size()) != 4:
            image = image.unsqueeze(0)
        image = self.preprocess(image)
        logits = self.model(image)
        self.num_queries += image.size(0)
        return logits

    def predict_label(self, image):
        logits = self.predict_prob(image)
        _, predict = torch.max(logits, 1)
        return predict

    def jpeg_compression(self, imgs):
        def to_jpeg(im):
            assert torch.is_tensor(im)
            im = ToPILImage()(im)
            savepath = BytesIO()
            im.save(savepath, 'JPEG', quality=75)
            im = Image.open(savepath)
            im = ToTensor()(im)
            return im
        jpeg_imgs = torch.zeros(imgs.shape).to(imgs.device)
        for i in range(imgs.shape[0]):
            jpeg_imgs[i] = to_jpeg(imgs[i].cpu()).to(imgs.device)
        return jpeg_imgs

    def gaussian_filter(self, imgs, sigma=2):
        def to_guass(imgs, sigma):
            output_imgs = np.zeros((imgs.shape), dtype=np.float32)
            for i in range(imgs.shape[0]):
                for j in range(3):
                    output_imgs[i, j, :, :] = ndimage.filters.gaussian_filter(imgs[i, j, :, :], sigma)
            return output_imgs
        guass_x = to_guass(imgs.detach().cpu().numpy(), sigma)
        return torch.tensor(guass_x).to(imgs.device)

    def deactivate_defense(self):
        self.defense_method = None

    def activate_defense(self, method, sigma=2):
        if method in ['jepg', 'guass']:
            self.defense_method = method
            self.guass_sigma = sigma
        else:
            raise NameError('False defense method')


