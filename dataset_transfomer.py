"""
This cli will transform the images directory from the structure of:
|-- final_submission
|   |-- 1
|   |-- 2
|   |-- ...
|   |-- n
|-- web
|   |-- 1_1.png
|   |-- 1_2.png
|   |-- ...
|   |-- n_n.png
|-- ai
|   |-- 1_1.png
|   |-- 1_2.png
|   |-- ...
|   |-- n_n.png

to the structure of:
|-- groups
|   |-- 1
|   |   |-- final
|   |   |   |-- 1.png
|   |   |   |-- ...
|   |   |   |-- n.png
|   |   |-- web
|   |   |   |-- 1_1.png
|   |   |   |-- ...
|   |   |   |-- 1_n.png
|   |   |-- ai
|   |   |   |-- 1_1.png
|   |   |   |-- ...
|   |   |   |-- 1_n.png
|   |-- 2
...etc
"""

import os
import shutil
import argparse
import re
import sys

parser = argparse.ArgumentParser(description='Transform the images directory.')
parser.add_argument('-f', '--final', help='Final submission directory', required=True)
parser.add_argument('-w', '--web', help='Web directory', required=True)
parser.add_argument('-a', '--ai', help='AI directory', required=True)
parser.add_argument('-o', '--output', help='Output directory (optional default is ./groups)', required=False, default='groups')

#Display help if no arguments are passed
if len(sys.argv)==1:
    parser.print_help(sys.stderr)
    sys.exit(1)

args = parser.parse_args()

final_dir = args.final
web_dir = args.web
ai_dir = args.ai
output_dir = args.output

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
groups = os.listdir(final_dir)

for group in groups:
    group_filename_pattern = r"^" + f"{group}" + r"[a-zA-Z]?_.+\.[a-zA-Z]+$"
    #Copy final images
    os.makedirs(os.path.join(output_dir, group))
    #os.makedirs(os.path.join(output_dir, group, 'final'))
    shutil.copytree(os.path.join(final_dir, group), os.path.join(output_dir, group, 'final'))
    
    #Copy web images
    group_web_images = [x for x in os.listdir(web_dir) if re.match(group_filename_pattern, x)]
    group_web_images_fullpath = [os.path.join(web_dir, x) for x in group_web_images]
    
    os.makedirs(os.path.join(output_dir, group, 'web'))
    for img in group_web_images_fullpath:
        shutil.copy(img, os.path.join(output_dir, group, 'web'))
        
    #Copy ai images
    group_ai_images = [x for x in os.listdir(ai_dir) if re.match(group_filename_pattern, x)]
    group_ai_images_fullpath = [os.path.join(ai_dir, x) for x in group_ai_images]
    
    os.makedirs(os.path.join(output_dir, group, 'ai'))
    for img in group_ai_images_fullpath:
        shutil.copy(img, os.path.join(output_dir, group, 'ai'))
        
print('Done!')