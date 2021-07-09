# ffmpeg -i video_00010.mp4 -filter:v fps=16 -ss 00:00:1.0 -t 2 out.mp4
#ffmpeg -i H:\Dog_Selected\video_00010.mp4 -filter:v fps=16 -ss 00:00:1.0 -t 2 G:\Google_Drive\Research\audio2video\a2v\trimmed_dataset\1.mp4
import skvideo.io
import os
import glob
from moviepy.editor import *

# cmd = "ffmpeg -i H:\\Dog_Selected\\video_00010.mp4 -filter:v fps=16 -ss 00:00:1.0 -t 2 G:\\Google_Drive\Research\\audio2video\\a2v\\trimmed_dataset\\2.mp4"
# os.system(cmd)
src = "H:\\Snoring_selected"
dest = "G:\\Google_Drive\Research\\audio2video\\a2v\\trimmed_dataset"

files = glob.glob(os.path.join(src, '*.mp4'))
i = 0
for file in files:
    cmd = "ffmpeg -i "+os.path.join(dest, file)+" -filter:v fps=16 -ss 00:00:1.0 -t 2 "+os.path.join(dest, str(i))+".mp4"
    i = i+1
    #print(cmd)
    os.system(cmd)
