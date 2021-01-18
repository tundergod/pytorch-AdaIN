# Style Transfer - Image
Our implementation of style transformation for image is based on [this work](https://github.com/naoto0804/pytorch-AdaIN?fbclid=IwAR0xkGfGRts96b_qyR_kJSBdEWUitJ-zfcOMN84jfLcyBD5pw7SaW6KnQCU). 
For better output result, we work on content-color preserving, luminance-style transferring, and spatial controlling.

## Associated files
- my_test.py

## Reproducing results
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

## Trying other images
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
