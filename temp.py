from skew_detect import SkewDetect
from deskew import Deskew
from text_cleaner import clean_image

img_path = './images/12.png'
clean_image(img_path, './')
img_path = 'cleaned.png'
#sd = SkewDetect(input_file=img_path, output_file='./output.txt', display_output='Yes')
#sd.run()

d = Deskew(input_file=img_path, display_image=False, output_file='output_img.png', r_angle=0)
d.run()