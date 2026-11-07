YOUR OWN TEST IMAGES -- one folder per CIFAR-10 class
=====================================================
Drop any photo (JPG / PNG / WebP / BMP, ANY size or aspect ratio) into the
folder matching what the photo shows, then refresh the web page
(http://127.0.0.1:8080) -- your images appear in section "2b - Your own
images" and can be clicked to run a prediction.

Images downloaded from Google work fine: the app resizes everything to
224x224 with the official CLIP preprocessing inside the code
(demo/app.py -> clip_image_transform), so no manual resizing is needed.
