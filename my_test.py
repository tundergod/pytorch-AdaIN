import argparse
from pathlib import Path

from skimage.exposure import cumulative_distribution
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F

import net
from function import adaptive_instance_normalization, coral
import matplotlib.pyplot as plt

def cdf(im):
 '''
 computes the CDF of an image im as 2D numpy ndarray
 '''
 c, b = cumulative_distribution(im) 
 # pad the beginning and ending pixels and their CDF values
 c = np.insert(c, 0, [0]*b[0])
 c = np.append(c, [1]*(255-b[-1]))
 return c

def hist_matching(c, c_t, im):
 '''
 c: CDF of input image computed with the function cdf()
 c_t: CDF of template image computed with the function cdf()
 im: input image as 2D numpy ndarray
 returns the modified pixel values
 ''' 
 pixels = np.arange(256)
 # find closest pixel-matches corresponding to the CDF of the input image, given the value of the CDF H of   
 # the template image at the corresponding pixels, s.t. c_t = H(pixels) <=> pixels = H-1(c_t)
 new_pixels = np.interp(c, c_t, pixels) 
 im = (np.reshape(new_pixels[im.ravel()], im.shape)).astype(np.uint8)
 return im

def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


def style_transfer(vgg, decoder, content, style, alpha=1.0,
                   interpolation_weights=None):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    style_f = vgg(style)
    if interpolation_weights:
        _, C, H, W = content_f.size()
        feat = torch.FloatTensor(1, C, H, W).zero_().to(device)
        base_feat = adaptive_instance_normalization(content_f, style_f)
        for i, w in enumerate(interpolation_weights):
            feat = feat + w * base_feat[i:i + 1]
        content_f = content_f[0:1]
    else:
        feat = adaptive_instance_normalization(content_f, style_f)
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat)

def spatial_control(vgg, decoder, content, stye, mask):
    content_f = vgg(content)
    style_f = vgg(style)#;import pdb; pdb.set_trace()
    mask = F.interpolate(mask, (content_f.size(2), content_f.size(3)))
    base_feat = adaptive_instance_normalization(content_f, style_f)
    feat = torch.sum(mask[:,0:1,:,:] * base_feat, dim=0, keepdim=True)
    return decoder(feat)


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content', type=str,
                    help='File path to the content image')
parser.add_argument('--content_dir', type=str,
                    help='Directory path to a batch of content images')
parser.add_argument('--style', type=str,
                    help='File path to the style image, or multiple style \
                    images separated by commas if you want to do style \
                    interpolation or spatial control')
parser.add_argument('--style_dir', type=str,
                    help='Directory path to a batch of style images')
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')
parser.add_argument('--decoder', type=str, default='models/decoder.pth')

# Additional options
parser.add_argument('--content_size', type=int, default=512,
                    help='New (minimum) size for the content image, \
                    keeping the original size if set to 0')
parser.add_argument('--style_size', type=int, default=512,
                    help='New (minimum) size for the style image, \
                    keeping the original size if set to 0')
parser.add_argument('--crop_content', action='store_true',
                    help='do center crop to create squared image')
parser.add_argument('--crop_style', action='store_true',
                    help='do center crop to create squared image')
parser.add_argument('--save_ext', default='.jpg',
                    help='The extension name of the output image')
parser.add_argument('--output', type=str, default='output',
                    help='Directory to save the output image(s)')

# Advanced options
parser.add_argument('--preserve_color', action='store_true',
                    help='If specified, preserve color of the content image')
parser.add_argument('--luminance_only', action='store_true',
                    help='If specified, perform style transfer \
                                    only in the luminance channel')
parser.add_argument('--alpha', type=float, default=1.0,
                    help='The weight that controls the degree of \
                             stylization. Should be between 0 and 1')
parser.add_argument(
    '--style_interpolation_weights', type=str, default='',
    help='The weight for blending the style of multiple style images')
parser.add_argument('--mask', type=str, default='',
                    help='The masks for spatial control')

args = parser.parse_args()

do_interpolation = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

output_dir = Path(args.output)
output_dir.mkdir(exist_ok=True, parents=True)

# Either --content or --contentDir should be given.
assert (args.content or args.content_dir)
if args.content:
    content_paths = [Path(args.content)]
else:
    content_dir = Path(args.content_dir)
    content_paths = [f for f in content_dir.glob('*')]

# Either --style or --styleDir should be given.
assert (args.style or args.style_dir)
if args.style:
    style_paths = args.style.split(',')
    if len(style_paths) == 1:
        style_paths = [Path(args.style)]
    else:
        if args.style_interpolation_weights != '':
            weights = [int(i) for i in args.style_interpolation_weights.split(',')]
            interpolation_weights = [w / sum(weights) for w in weights]
            do_interpolation = True
else:
    style_dir = Path(args.style_dir)
    style_paths = [f for f in style_dir.glob('*')]

if args.mask:
    mask_paths = args.mask.split(',')
    assert len(style_paths) == len(mask_paths)

decoder = net.decoder
vgg = net.vgg

decoder.eval()
vgg.eval()

decoder.load_state_dict(torch.load(args.decoder))
vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:31])

vgg.to(device)
decoder.to(device)

content_tf = test_transform(args.content_size, args.crop_content)
style_tf = test_transform(args.style_size, args.crop_style)
for content_path in content_paths:
    if do_interpolation:  # one content image, N style image
        content = content_tf(Image.open(str(content_path)))
        content = torch.stack([content for p in style_paths])
        style = [style_tf(Image.open(str(p))) for p in style_paths]
        if args.preserve_color:
            style = [coral(s, content) for s in style]
        style = torch.stack(style)
        
        style = style.to(device)
        content = content.to(device)
        with torch.no_grad():
            output = style_transfer(vgg, decoder, content, style,
                                    args.alpha, interpolation_weights)
        output = output.cpu()
        output_name = output_dir / '{:s}_interpolation{:s}'.format(
            content_path.stem, args.save_ext)
        save_image(output, str(output_name))

    elif args.mask:
        content = Image.open(str(content_path))
        if args.luminance_only:
            content = content.convert('HSV')
            content_pil = content
            content = np.array(content)
            content_cdf = cdf(content[:,:,2])
            content = Image.fromarray(content[:,:,2])
            content = content.convert('RGB')
        content = content_tf(content)
        content = torch.stack([content for p in style_paths])

        styles = []
        for p in style_paths:
            style = Image.open(p)
            if args.luminance_only:
                style = style.convert('HSV')
                style = np.array(style)
                style_cdf = cdf(style[:,:,2])
                style[:,:,2] = hist_matching(style_cdf, content_cdf, style[:,:,2])
                style = Image.fromarray(style[:,:,2])
                style = style.convert('RGB')
            style = style_tf(style)
            if args.preserve_color:
                style = coral(style, content[0])
            styles.append(style)
        style = torch.stack(styles)

        mask = [content_tf(Image.open(str(p))) for p in mask_paths]
        mask = torch.stack(mask)

        style = style.to(device)
        content = content.to(device)
        mask = mask.to(device)
        if args.luminance_only:
            content_lum = content.to(device)
            style_lum = style.to(device)
            with torch.no_grad():
                output_lum = spatial_control(vgg, decoder, 
                                             content_lum, style_lum, mask)
            output_lum = output_lum.cpu().squeeze()[0]
            output = content
            if (output_lum.size(-1) != output.size(-1)):
                pad = (output_lum.size(-1) - output.size(-1)) // 2
                output_lum = output_lum[:, pad:-pad]
        else:
            with torch.no_grad():
                output = spatial_control(vgg, decoder, 
                                         content, style, mask)
            output = output.cpu()
        output_name = output_dir / '{:s}_spatial{:s}'.format(
            content_path.stem, args.save_ext)
        if args.luminance_only:
            output_lum = output_lum.numpy()
            minimum, maximum = output_lum.min(), output_lum.max()
            output_lum = (output_lum - minimum) / (maximum - minimum)
            output_lum = (output_lum * 255).astype(np.uint8)
            content_pil = content_pil.resize((output_lum.shape[1], 
                                              output_lum.shape[0]))
            content_np = np.array(content_pil)
            content_np[:,:,2] = output_lum
            output = Image.fromarray(content_np, mode='HSV')
            output = output.convert('RGB')
            output.save(str(output_name))
        else:
            save_image(output, str(output_name))


    else:  # process one content and one style
        for style_path in style_paths:
            content = Image.open(str(content_path))
            if args.luminance_only:
                content = content.convert('HSV')
                content_pil = content
                content = np.array(content)
                content_cdf = cdf(content[:,:,2])
                content = Image.fromarray(content[:,:,2])
                content = content.convert('RGB')
            content = content_tf(content)

            style = Image.open(str(style_path))
            if args.luminance_only:
                style = style.convert('HSV')
                style = np.array(style)
                style_cdf = cdf(style[:,:,2])
                style[:,:,2] = hist_matching(style_cdf, content_cdf, style[:,:,2])
                style = Image.fromarray(style[:,:,2])
                style = style.convert('RGB')
            style = style_tf(style)

            if args.preserve_color:
                style = coral(style, content)
            if args.luminance_only:
                content_lum = content.to(device).unsqueeze(0)
                style_lum = style.to(device).unsqueeze(0)
                with torch.no_grad():
                    output_lum = style_transfer(vgg, decoder, content_lum, style_lum,
                                            args.alpha)
                output_lum = output_lum.cpu().squeeze()[0]
                output = content
                if (output_lum.size(-1) != output.size(-1)):
                    pad = (output_lum.size(-1) - output.size(-1)) // 2
                    output_lum = output_lum[:, pad:-pad]
            else:
                style = style.to(device).unsqueeze(0)
                content = content.to(device).unsqueeze(0)
                with torch.no_grad():
                    output = style_transfer(vgg, decoder, content, style,
                                            args.alpha)
                output = output.cpu()

            output_name = output_dir / '{:s}_stylized_{:s}{:s}'.format(
                content_path.stem, style_path.stem, args.save_ext)
            if args.luminance_only:
                output_lum = output_lum.numpy()
                minimum, maximum = output_lum.min(), output_lum.max()
                output_lum = (output_lum - minimum) / (maximum - minimum)
                output_lum = (output_lum * 255).astype(np.uint8)
                content_pil = content_pil.resize((output_lum.shape[1], 
                                                output_lum.shape[0]))
                content_np = np.array(content_pil)
                content_np[:,:,2] = output_lum
                output = Image.fromarray(content_np, mode='HSV')
                output = output.convert('RGB')
                output.save(str(output_name))
            else:
                save_image(output, str(output_name))
