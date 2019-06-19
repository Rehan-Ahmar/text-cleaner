import os
import argparse
from text_cleaner import clean_image
from skew_corrector import process_image

def start():
    parser = argparse.ArgumentParser(description='Text Cleaner and Skew Corrector')
    parser.add_argument('--indir', type=str, default='./images/', help='Input directory path')
    parser.add_argument('--outdir', type=str, default='./outputs-skew2/', help='Output directory path')
    args = parser.parse_args()
    
    for filename in os.listdir(args.indir):
        if filename.lower().endswith('.jpg') or filename.lower().endswith('.png'):
            img_path = os.path.join(args.indir, filename)
            output_path = os.path.join(args.outdir, os.path.splitext(filename)[0])
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            clean_image(img_path, output_path)
            cleaned_img_path = os.path.join(output_path, 'cleaned.png')
            process_image(cleaned_img_path, output_path)

if __name__ == '__main__':
    start()