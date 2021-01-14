######## woman ########
# 1. oil
python my_test.py \
--content imgs/content/woman/woman.jpg \
--style \
imgs/style/woman/oil.png,\
imgs/style/woman/oil.png,\
imgs/style/woman/oil.png \
--style_size 400 \
--crop_style \
--content_size 0 \
--mask \
imgs/content/woman/woman_eyes.jpg,\
imgs/content/woman/woman_face_hair.jpg,\
imgs/content/woman/woman_background.jpg \
--output imgs/reproduce/woman/oil \
--alpha 0,0.6,0.8 \
--preserve_color

# 2. water
python my_test.py \
--content imgs/content/woman/woman.jpg \
--style \
imgs/style/woman/water.jpg,\
imgs/style/woman/water.jpg,\
imgs/style/woman/water.jpg \
--style_size 400 \
--crop_style \
--content_size 0 \
--mask \
imgs/content/woman/woman_eyes.jpg,\
imgs/content/woman/woman_face_hair.jpg,\
imgs/content/woman/woman_background.jpg \
--output imgs/reproduce/woman/water \
--alpha 0.1,0.8,1 \
--preserve_color

# 3. sketch
python my_test.py \
--content imgs/content/woman/woman_bw.jpg \
--style \
imgs/style/woman/sketch_front.jpg,\
imgs/style/woman/sketch_front.jpg,\
imgs/style/woman/sketch_back.jpg \
--style_size 400 \
--crop_style \
--content_size 0 \
--mask \
imgs/content/woman/woman_eyes.jpg,\
imgs/content/woman/woman_face_hair.jpg,\
imgs/content/woman/woman_background.jpg \
--output imgs/reproduce/woman/sktech \
--alpha 0,0.5,1

# 4. ink
python my_test.py \
--content imgs/content/woman/woman_bw.jpg \
--style \
imgs/style/woman/ink_front.jpg,\
imgs/style/woman/ink_front.jpg,\
imgs/style/woman/ink_back.jpg \
--style_size 400 \
--crop_style \
--content_size 0 \
--mask \
imgs/content/woman/woman_eyes.jpg,\
imgs/content/woman/woman_face_hair.jpg,\
imgs/content/woman/woman_background.jpg \
--output imgs/reproduce/woman/ink \
--alpha 0.2,0.8,0.8