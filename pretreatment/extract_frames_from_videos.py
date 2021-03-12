import os
import sys
import subprocess
from tqdm import tqdm
import pickle
import shutil
import argparse
import glob

def extract_frames(video, dst, suffix, strategy, fps, vframes):
    with open(os.devnull, "w") as ffmpeg_log:
        if os.path.exists(dst):
            print(" cleanup: " + dst + "/")
            shutil.rmtree(dst)
        os.makedirs(dst)
        if strategy == 0:
            video_to_frames_command = ["ffmpeg",
                                       '-y',
                                       '-i', video,  # input file
                                       '-vf', "scale=iw:-1", # input file
                                       '{0}/%05d.{1}'.format(dst, suffix)]
        else:
            video_to_frames_command = ["ffmpeg",
                                       '-y',
                                       '-i', video,  # input file
                                       '-vf', "scale=iw:-1", # input file
                                       '-r', fps, #fps 5
                                       '-vframes', vframes,
                                       '{0}/%05d.{1}'.format(dst, suffix)]
        subprocess.call(video_to_frames_command,
                        stdout=ffmpeg_log, stderr=ffmpeg_log)

def run(params):
    if params['info_path']:
        vid2id = pickle.load(open(params['info_path'], 'rb'))['info']['vid2id']
        id2vid = {v: k for k, v in vid2id.items()}
    else:
        id2vid = None

    video_path_list = glob.glob(os.path.join(params['video_path'], '*.%s' % params['video_suffix']))
    tqdm.write('There are %d .%s files' % (len(video_path_list), params['video_suffix']))

    for video_path in tqdm(video_path_list):
        video_id, ext = os.path.splitext(video_path.split('/')[-1])
        if id2vid is not None:
            video_id = id2vid[video_id]

        dst_frame_path = os.path.join(params['frame_path'], video_id)

        extract_frames(
                video=video_path, 
                dst=dst_frame_path, 
                suffix=params['frame_suffix'], 
                strategy=params['strategy'], 
                fps=params['fps'], 
                vframes=params['vframes']
            )


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True, help='the path to load all the videos')
    parser.add_argument("--frame_path", type=str, required=True, help='the path to put the extracted frames')

    parser.add_argument("--info_path", type=str, default='', help='mapping the video name to the video_id')
    parser.add_argument("--strategy", type=int, default=0, help='0: extract all the frames; 1: need to specify fps and vframes')
    parser.add_argument("--fps", type=str, default='5', help='the number of frames you want to extract within 1 second')
    parser.add_argument("--vframes", type=str, default='60', help='the maximun number of frames you want to extract')

    parser.add_argument("--video_suffix", type=str, default='mp4')
    parser.add_argument("--frame_suffix", type=str, default='jpg')

    args = parser.parse_args()
    params = vars(args)

    assert os.path.exists(params['video_path'])
    if not os.path.exists(params['frame_path']):
        os.makedirs(params['frame_path'])

    if params['strategy'] == 0:
        print_info = 'all frames from a video'
    else:
        print_info = 'fps=%s, vframes=%s' % (params['fps'], params['vframes'])
    tqdm.write('Extraction strategy: %s' % print_info)

    run(params)

