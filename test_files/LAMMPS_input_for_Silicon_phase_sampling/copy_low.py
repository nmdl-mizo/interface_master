import os
import shutil

os.mkdir('low_file_dir')

with open('low_files','r') as f:
    lines = f.readlines()

home_path = os.getcwd()
copy_path = os.path.join(home_path, 'low_file_dir')
for i in range(len(lines)):
    work_dir = lines[i].split()[0]
    dump_index = 'dump' + lines[i].split()[1]
    here = os.path.join(home_path, work_dir)
    here = os.path.join(here, dump_index)
    there = os.path.join(copy_path, 'dump'+str(i))
    shutil.copy(here, there)
