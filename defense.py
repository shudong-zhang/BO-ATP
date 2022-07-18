from torchvision.transforms import ToPILImage, ToTensor
from io import BytesIO
from PIL import Image
import torch

def _jpeg_compression(im):
    assert torch.is_tensor(im)
    im = ToPILImage()(im)
    savepath = BytesIO()
    im.save(savepath, 'JPEG', quality=75)
    im = Image.open(savepath)
    im = ToTensor()(im)
    return im
def jpeg_compression_batch(imgs):
    jpeg_imgs = torch.zeros(imgs.shape).to(imgs.device)
    for i in range(imgs.shape[0]):
        jpeg_imgs[i] = _jpeg_compression(imgs[i].cpu()).to(imgs.device)
    return jpeg_imgs