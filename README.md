# pytorch-AdaIN
# CSIE 5612: Digital Image Processing (DIP) Final Project (Style Transfer on Image and Video)

This is a final project from the Digital Image Processing (DIP) Course (CSIE 5612) on Fall 2020 in National Taiwan University (NTU), Taiwan. This project aims for rendering images and videos with four required styles: sketch, ink painting (ink), watercolor (water), and oil painting (oil). This project is implemented based on a forked version of an unofficial pytorch implementation of a paper, Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization [Huang+, ICCV2017].
We are really grateful to the both [pytorch implementation](https://github.com/naoto0804/pytorch-AdaIN) and [original implementation](https://github.com/xunhuang1995/AdaIN-style) in Torch by the authors, which is very useful.

## Requirements
Please install requirements by `pip install -r requirements.txt`

- Python 3.5+
- PyTorch 0.4+
- TorchVision
- Pillow

(optional, for training)
- tqdm
- TensorboardX

## Usage

### Download models
Download [vgg_normalized.pth](https://drive.google.com/file/d/1bMfhMMwPeXnYSQI6cDWElSZxOxc6aVyr/view?usp=sharing)/[decoder.pth](https://drive.google.com/file/d/1EpkBA2K2eYILDSyPTt0fztz59UjAIpZU/view?usp=sharing) and put them under `models/`.

### Test
Use `--content` and `--style` to provide the respective path to the content and style image.
```
CUDA_VISIBLE_DEVICES=<gpu_id> python test.py --content input/content/cornell.jpg --style input/style/woman_with_hat_matisse.jpg
```

You can also run the code on directories of content and style images using `--content_dir` and `--style_dir`. It will save every possible combination of content and styles to the output directory.
```
CUDA_VISIBLE_DEVICES=<gpu_id> python test.py --content_dir input/content --style_dir input/style
```

This is an example of mixing four styles by specifying `--style` and `--style_interpolation_weights` option.
```
CUDA_VISIBLE_DEVICES=<gpu_id> python test.py --content input/content/avril.jpg --style input/style/picasso_self_portrait.jpg,input/style/impronte_d_artista.jpg,input/style/trial.jpg,input/style/antimonocromatismo.jpg --style_interpolation_weights 1,1,1,1 --content_size 512 --style_size 512 --crop
```

Some other options:
* `--content_size`: New (minimum) size for the content image. Keeping the original size if set to 0.
* `--style_size`: New (minimum) size for the content image. Keeping the original size if set to 0.
* `--alpha`: Adjust the degree of stylization. It should be a value between 0.0 and 1.0 (default).
* `--preserve_color`: Preserve the color of the content image.


### Train
Use `--content_dir` and `--style_dir` to provide the respective directory to the content and style images.
```
CUDA_VISIBLE_DEVICES=<gpu_id> python train.py --content_dir <content_dir> --style_dir <style_dir>
```

For more details and parameters, please refer to --help option.

I share the model trained by this code [here](https://drive.google.com/file/d/1YIBRdgGBoVllLhmz_N7PwfeP5V9Vz2Nr/view?usp=sharing)

## Style Transfer on Image
For better output result, we work on content-color preserving, luminance-style transferring, and spatial controlling.

### Associated files
- my_test.py

### Reproducing results
The results in the presentation and all of the necessary content/style images can be found in the folder [img](https://github.com/tundergod/pytorch-AdaIN/tree/master/imgs).
To reproduce all the results, run
```bash
./reproduce.sh
```
or run
```bash
./lake.sh
./woman.sh
./street.sh
```
respectively to reproduce results for each image.

### Trying other images
Use `--content` and `--style` to provide the respective path to the content and style image.
```bash
python my_test.py --content imgs/content/woman/woman.jpg --style imgs/style/woman/oil.png
```
Some other options:
* `--content_size`: New (minimum) size for the content image. Keeping the original size if set to 0.
* `--style_size`: New (minimum) size for the content image. Keeping the original size if set to 0.
* `--alpha`: Adjust the degree of stylization. It should be a value between 0.0 and 1.0 (default).
* `--preserve_color`: Preserve the color of the content image by histogram-matching technique.
* `--luminance_only`: Preserve the color of the content image by luminance-transfer technique.

Spatial controlling:

Use `--mask` to provide the respective path to mask image.
Providing different `--style` and `--alpha` with respect to the masks to stylize different parts of the image separately.
```bash
python my_test.py \
--content imgs/content/lake/lake_bw.jpg \
--style imgs/style/lake/ink_front.jpg,imgs/style/lake/ink_back.jpg \
--mask imgs/content/lake/lake_foreground.jpg,imgs/content/lake/lake_background.jpg \
--alpha 0.8,1 \
--style_size 250 \
--crop_style \
--content_size 0 \
--output imgs/reproduce/lake/ink
```

## Style Transfer on Video
For better output result, we work on content-color preserving, luminance-style transferring, and video deflickering.

## Associated files
- my_video.py
- luminance_adjustment.py
- temporal_smoothing.py

## Style transfer
### Baseline
Use `--style` to select the desired style: oil/water/sketch/ink.
```bash
python my_video.py --style [style options]
```
By default, in the procedure of style tranfer for *sketch* and *ink*, we will convert the contents into graysacle and the grayscale contents will be stylized into outputs with *all information, i.e., luminance and color* in the style image.
For the styles *oil* and *water*, we will only transfer the style of *luminance* of the contents.
Additional argument `--UV_color` can be specified if we want to transfer *oil* and *water* with *all information* in the style image and preserve the color of contents after stylization.
Output file(s):
(Output videos can be found in the directory */output*)

- **[*specified_style*].mp4:** (for *sketch* and *ink*)
The input video `ntu.mp4` is style transferred into the style specified.
The color of output video depends on the style image.

- **luma_[*specified_style*].mp4:** (for *oil* and *water*)
The input video `ntu.mp4` is style transferred into the style specified.
Style transfer is only applyed on the luminance dimension.

- **UV_[*specified_style*].mp4:** (for *oil* and *water*)
The input video `ntu.mp4` is style transferred into the style specified.
Style transfer is achieved through being stylized by whole information in the style image and the U and V values of contents are preserved.

Note that these output files are with flicker.
To perform video deflickering, follow the instructions below.

## Video deflickering
### Temporal smoothing:
```bash
python temporal_smoothing.py [videoPath]
```
Output file(s):
(Output videos can be found in the directory */output*)

- **TS_[videoName].mp4:**

### Luminance adjustment: (suggested)
```bash
python luminance_adjustment.py [videoPath]
```
Output file(s):
(Output videos can be found in the directory */output*)
- **LA_[videoName].mp4:**

## References
- [1]: X. Huang and S. Belongie. "Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization.", in ICCV, 2017.
- [2]: [Original implementation in Torch](https://github.com/xunhuang1995/AdaIN-style)
