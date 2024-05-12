from unittest import mock
from simpleocr.files import open_image
from simpleocr.grounding import UserGrounder
from simpleocr.opencv_utils import draw_classes
from simpleocr.segmentation import ContourSegmenter

import cv2

# temp_img = cv2.imread('./simpleocr/data/image_01.png')
# cv2.imwrite('./simpleocr/data/image_01.png', 255- temp_img)

img = open_image("<test_file_path>")  # relative or absolute

segmenter = ContourSegmenter()
segments = segmenter.process(img.image)
# segmenter.display()
terminal = UserGrounder()
characters = "0" * len(segments)
mock_input_gen = iter(characters)


terminal.ground(img, segments)
img.set_ground(img.ground.segments, img.ground.classes, write_file=True)
