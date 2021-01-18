# Style Transfer - Video
Our implementation of style transformation for video is based on [this work](https://github.com/naoto0804/pytorch-AdaIN?fbclid=IwAR0xkGfGRts96b_qyR_kJSBdEWUitJ-zfcOMN84jfLcyBD5pw7SaW6KnQCU).
For better output result, we work on content-color preserving, luminance-style transferring, and video deflickering.

## Associated files
- my_video.py
- luminance_adjustment.py
- temporal_smoothing.py

## Style transfer
### Baseline
Use `--style` to select the desired style: oil/water/sketch/ink.
```bash
python3 my_video.py --style [style options]
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
Temporal smoothing:
```bash
python3 temporal_smoothing.py [videoPath]
```
Output file(s):  
(Output videos can be found in the directory */output*)

- **TS_[videoName].mp4:**


Luminance adjustment: (suggested)
```bash
python3 luminance_adjustment.py [videoPath]
```
Output file(s):  
(Output videos can be found in the directory */output*)
- **LA_[videoName].mp4:**

