tyle Transfer - Video
Our implementation of style transformation for video is based on [this work](https://github.com/naoto0804/pytorch-AdaIN?fbclid=IwAR0xkGfGRts96b_qyR_kJSBdEWUitJ-zfcOMN84jfLcyBD5pw7SaW6KnQCU).
For better output result, we work on content-color preserving, luminance-style transferring, and video deflickering.

## Associated files
- my_video.py
- my_video_lumi.py
- deflicker_temporal.py
- deflicker_color.py

## Style image
The work supports style tranfterring into four styles: oil, water, sketch, and ink.

## Style transfer
### Baseline
Perform Style transferring with *all information*, including *color and luminance*, of style image.
Use `--style` to select desired style: oil/water/sketch/ink.
```bash
python3 my_video.py --style [style options]
```
Output file(s):
(Output videos can be found in the directory *output*)
- **ntu_stylized_[*specified_style*].mp4:**
The input video `ntu.mp4` is style transferred into the style specified. The color of output video depends on the style image.
- **ntu_stylized_color_preserved_[*specified_style*].mp4:**
The input video `ntu.mp4` is style transferred into the style specified. The color of output video is applied with the chrominance and the chroma of input video.
In our implementation, this file will be generated only if the specified style is *oil* or *water* to provide closer images to real world visualization. 

### Baseline - Style transfer on luminance
Perform style transferring with only *the luminance information* of style image.
Use `--style` to select desired style: oil/water/sketch/ink.
```bash
python3 my_video.py --style [style options]
```
Output file(s):
(Output videos can be found in the directory *output*)
- **lumi_ntu_stylized_[*specified_style*].mp4:**
The input video `ntu.mp4` is style transferred into the style specified on the luminance dimension.


## Video deflickering
Temporal smoothing:
```bash
python3 deflicker_temporal.py [videoPath]
```

Luminance adjustance:
```bash
python3 deflicker_color.py [videoPath]
```

## References
[1] 
