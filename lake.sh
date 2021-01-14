######## lake ########
# 1. oil
python my_test.py \
--content imgs/content/lake/lake.JPG \
--style \
imgs/style/lake/oil.jpg,\
imgs/style/lake/oil.jpg \
--style_size 250 \
--crop_style \
--content_size 0 \
--mask \
imgs/content/lake/lake_foreground.jpg,\
imgs/content/lake/lake_background.jpg \
--output imgs/reproduce/lake/oil \
--alpha 1,0.5 \
--luminance_only

# 2. water
python my_test.py \
--content imgs/content/lake/lake.JPG \
--style \
imgs/style/lake/water.jpg,\
imgs/style/lake/water.jpg \
--style_size 250 \
--crop_style \
--content_size 0 \
--mask \
imgs/content/lake/lake_foreground.jpg,\
imgs/content/lake/lake_background.jpg \
--output imgs/reproduce/lake/water \
--alpha 0.5,0.2 \
--luminance_only

# 3. sketch
python my_test.py \
--content imgs/content/lake/lake_bw.jpg \
--style \
imgs/style/lake/sketch_front.jpg,\
imgs/style/lake/sketch_back.jpg \
--style_size 250 \
--crop_style \
--content_size 0 \
--mask \
imgs/content/lake/lake_foreground.jpg,\
imgs/content/lake/lake_background.jpg \
--output imgs/reproduce/lake/sketch \
--alpha 0.5,1 

# 4. ink
python my_test.py \
--content imgs/content/lake/lake_bw.jpg \
--style \
imgs/style/lake/ink_front.jpg,\
imgs/style/lake/ink_back.jpg \
--style_size 250 \
--crop_style \
--content_size 0 \
--mask \
imgs/content/lake/lake_foreground.jpg,\
imgs/content/lake/lake_background.jpg \
--output imgs/reproduce/lake/ink \
--alpha 0.8,1