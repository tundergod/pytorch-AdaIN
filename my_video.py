import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
import argparse
from pathlib import Path
from tqdm import tqdm
from skimage.exposure import cumulative_distribution
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2
import imageio
from torchvision import transforms
from torchvision.utils import save_image

import net
from function import adaptive_instance_normalization, coral

import warnings
warnings.filterwarnings("ignore")

#-----CT
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
#-----CT

#--------------------------For preserving color, CLTsai------------------------------------#
def preserve(img, content_yuv):
    img = np.squeeze(img)
    #img = img[:,:,(2,1,0)]  # bgr to rgb
    if content_yuv is not None:
        yuv = cv2.cvtColor(np.float32(img), cv2.COLOR_RGB2YUV)
        yuv[:,:,1:3] = content_yuv[:,:,1:3]
        img = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
    #img = np.clip(img, 0, 255).astype(np.uint8)
    img = np.clip(img, 0, 255)
    return img
#------------------------------------------------------------------------------------------#


def test_transform():
    transform_list = []
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


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content_video', type=str, default='./input/videos/ntu.mp4',
                    help='File path to the content video')
parser.add_argument('--style', type=str,
                    help='Transferred style: oil, water, sketch, ink')
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')
parser.add_argument('--decoder', type=str, default='models/decoder.pth')

# Additional options
parser.add_argument('--save_ext', default='.mp4',
                    help='The extension name of the output video')
parser.add_argument('--output', type=str, default='output',
                    help='Directory to save the output image(s)')
parser.add_argument('--UV_color', action='store_true',
                    help='Preserving color by applying UV dimension of content')

# Advanced options
parser.add_argument('--alpha', type=float, default=0.7,
                    help='The weight that controls the degree of \
                             stylization. Should be between 0 and 1')

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

output_dir = Path(args.output)
output_dir.mkdir(exist_ok = True, parents = True)

# --content_video should be given.
assert (args.content_video)
if args.content_video:
    content_path = Path(args.content_video)

# --style should be given
assert (args.style)
if args.style=="oil":
    style_path = Path("./input/style/video_oil.jpg")
elif args.style=="water":
    style_path = Path("./input/style/video_water.jpg")
elif args.style=="sketch":
    style_path = Path("./input/style/video_sketch.jpg")
elif args.style=="ink":
    style_path = Path("./input/style/video_ink.jpg")
else:
    print("wrong style")
    exit(0)

decoder = net.decoder
vgg = net.vgg

decoder.eval()
vgg.eval()

decoder.load_state_dict(torch.load(args.decoder))
vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:31])

vgg.to(device)
decoder.to(device)

content_tf = test_transform()
style_tf = test_transform()
        
#get video fps & video size
content_video = cv2.VideoCapture(args.content_video)
fps = int(content_video.get(cv2.CAP_PROP_FPS))
content_video_length = int(content_video.get(cv2.CAP_PROP_FRAME_COUNT))
output_width = int(content_video.get(cv2.CAP_PROP_FRAME_WIDTH))
output_height = int(content_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

assert fps != 0, 'Fps is zero, Please enter proper video path'

pbar = tqdm(total = 362)        

if args.UV_color or args.style=="sketch" or args.style=="ink":

    output_video_path = output_dir / '{:s}{:s}'.format(
                args.style, args.save_ext)
    args.alpha = 0.6
    #--------------------------For preserving color, CLTsai------------------------------------#
    if args.style=="oil" or args.style=="water":
        output_video_path2 = output_dir / 'UV_{:s}{:s}'.format(
                    args.style, args.save_ext)
        writer_preserve = imageio.get_writer(output_video_path2, mode='I', fps=fps)
    #------------------------------------------------------------------------------------------#
    
    writer = imageio.get_writer(output_video_path, mode='I', fps=fps)
       
    style_img = Image.open(style_path)
    #while(True):
    while(content_video.isOpened()):
        ret, content_img = content_video.read()
        
        if not ret:
            break
        if args.style=="sketch" or args.style=="ink":
            content_img = cv2.cvtColor(content_img, cv2.COLOR_BGR2GRAY)
            content_img = cv2.cvtColor(content_img, cv2.COLOR_GRAY2BGR)

        cotent_img = cv2.cvtColor(content_img, cv2.COLOR_BGR2RGB)
        content = content_tf(Image.fromarray(content_img))
        style = style_tf(style_img)

        style = style.to(device).unsqueeze(0)
        content = content.to(device).unsqueeze(0)
        
    #--------------------------For preserving color, CLTsai------------------------------------#
        if args.style=="oil" or args.style=="water":
            content_yuv = cv2.cvtColor(np.float32(content_img), cv2.COLOR_RGB2YUV)
    #------------------------------------------------------------------------------------------#
        
        with torch.no_grad():
            output = style_transfer(vgg, decoder, content, style,
                                    args.alpha)
        output = output.cpu()
        output = output.squeeze(0)
        output = np.array(output)*255
        
        output = np.transpose(output, (1,2,0))
        output = np.clip(output, 0, 255)
    #--------------------------For preserving color, CLTsai------------------------------------#
        if args.style=="oil" or args.style=="water":
            output_preserve = preserve(output,content_yuv)
            writer_preserve.append_data(np.array(output_preserve))
    #------------------------------------------------------------------------------------------#
        writer.append_data(np.array(output))
        pbar.update(1)
    
    content_video.release()

else:
    output_video_path = output_dir / 'luma_{:s}{:s}'.format(
                args.style, args.save_ext)
    
    #--------------------------For preserving color, CLTsai------------------------------------#
    #if args.style=="oil" or args.style=="water":
        #output_video_path2 = output_dir / 'lumi_{:s}_stylized_color_preserved_{:s}{:s}'.format(
                    #content_path.stem, args.style, args.save_ext)
        #writer_preserve = imageio.get_writer(output_video_path2, mode='I', fps=fps)
    #------------------------------------------------------------------------------------------#
    
    writer = imageio.get_writer(output_video_path, mode='I', fps=fps)
    
    
    style_img = Image.open(style_path)
    style_img = style_img.convert('HSV')
    style_img = np.array(style_img)
    style_cdf = cdf(style_img[:,:,2])
    style_img = Image.fromarray(style_img[:,:,2])
    style_img_org = style_img.convert('RGB')

    
    if args.style=="sketch" or args.style=="ink":
        args.alpha = 0.6
    
    while(content_video.isOpened()):
        ret, content_img = content_video.read()
        
        if not ret:
            break
        if args.style=="sketch" or args.style=="ink":
            content_img = cv2.cvtColor(content_img, cv2.COLOR_BGR2GRAY)
            content_img = cv2.cvtColor(content_img, cv2.COLOR_GRAY2BGR)
            
    #--------------CT
        content_img_pil = Image.fromarray(cv2.cvtColor(content_img,cv2.COLOR_BGR2RGB))
        content = content_img_pil.convert('HSV')
        content_pil = content
        content = np.array(content)
        content_cdf = cdf(content[:,:,2])
        content = Image.fromarray(content[:,:,2])
        content = content.convert('RGB')
        content = content_tf(content)
        h, w = content.size(-2), content.size(-1)        

        style_img = style_img_org.convert('HSV')
        style_img = np.array(style_img)
        style_img[:,:,2] = hist_matching(style_cdf, content_cdf, style_img[:,:,2])
        style_img = Image.fromarray(style_img[:,:,2])
        style_img = style_img.convert('RGB')
        style = style_tf(style_img)
    #--------------CT
    
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
        output_lum = output_lum.numpy()
        minimum, maximum = output_lum.min(), output_lum.max()
        output_lum = (output_lum - minimum) / (maximum - minimum)
        output_lum = (output_lum * 255).astype(np.uint8)
        if output_lum.shape[1] > w:
            pad = (output_lum.shape[1] - w) // 2
            output_lum = output_lum[:,pad:-pad]
        if output_lum.shape[0] > h:
            pad = (output_lum.shape[0] - h) // 2
            output_lum = output_lum[pad:-pad,:]
        content_np = np.array(content_pil)
        content_np[:,:,2] = output_lum
        output = Image.fromarray(content_np, mode='HSV')
        output = output.convert('RGB')
        
        
    #--------------------------For preserving color, CLTsai------------------------------------#
        #if args.style=="oil" or args.style=="water":
            #content_yuv = cv2.cvtColor(np.float32(content_img_pil), cv2.COLOR_RGB2YUV)
    #------------------------------------------------------------------------------------------#
    #--------------------------For preserving color, CLTsai------------------------------------#
        #if args.style=="oil" or args.style=="water":
            #output_preserve = preserve(output,content_yuv)
            #writer_preserve.append_data(np.array(output_preserve))
    #------------------------------------------------------------------------------------------#
        writer.append_data(np.array(output))
        pbar.update(1)
    
    content_video.release()
