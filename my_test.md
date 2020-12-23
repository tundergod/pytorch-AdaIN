多加了兩個功能：`--luminance_only`和 `--mask`
* `--luminance_only`和原本就有的`--preserve_color`一樣，可以在style transfer時保有原本content image的colors  
不過`--preserve_color`是去match content跟style的color，  
而`--luminance_only`是只在content的luminance layer做style transfer，再加回content image上。  
兩者的效果不太一樣，可以都試試看。
* `--mask`則是做spatial control用的(詳參論文)，可以根據mask對不同位置給不同的style。  
用法大概像這樣：  
```
python my_test.py \
--content content.jpg \
--style style1.jpg,style2.jpg,style3.jpg \
--style_size 512 \
--crop_style \
--content_size 0 \
--mask mask1.jpg,mask2.jpg,mask3.jpg 
```
在`content.jpg`中，`style1`會作用的`mask1`白色的地方，`style2`對應`mask2`，以此類推。
