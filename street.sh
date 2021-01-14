######## street ########
# 1. oil
python my_test.py \
--content imgs/content/street/street.jpg \
--style imgs/style/street/oil.jpg \
--style_size 0 \
--content_size 0 \
--output imgs/reproduce/street/oil \
--luminance_only

# 2. ink
python my_test.py \
--content imgs/content/street/street_paint.jpg \
--style imgs/style/street/ink.jpg \
--style_size 0 \
--content_size 0 \
--alpha 0.5 \
--output imgs/reproduce/street/ink \