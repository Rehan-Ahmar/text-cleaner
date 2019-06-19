import os
import shutil
from tempfile import TemporaryDirectory
from pdf2image import convert_from_path

def convert_pdf_to_images(pdf_path, dpi, output_folder, fmt):
    pdf_name = os.path.basename(pdf_path)
    with TemporaryDirectory() as temp_dir:
        convert_from_path(pdf_path=pdf_path, dpi=dpi, output_folder=temp_dir, fmt=fmt, output_file=pdf_name)
        num_pages = len(os.listdir(temp_dir))
        for file_name in os.listdir(temp_dir):
            shutil.copy(os.path.join(temp_dir, file_name), output_folder)
    return num_pages
    
convert_pdf_to_images(r'pdfs/input1.pdf', 400, r'images/', 'png')
convert_pdf_to_images(r'pdfs/input2.pdf', 400, r'images/', 'png')