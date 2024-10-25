from model import BallTrackerNet
from bounce_detector import BounceDetector
import torch
import cv2
import csv
from general import postprocess
from tqdm import tqdm
import numpy as np
import argparse
from itertools import groupby
from scipy.spatial import distance


def read_video(path_video):
    """ Read video file    
    :params
        path_video: path to video file
    :return
        frames: list of video frames
        fps: frames per second
    """
    cap = cv2.VideoCapture(path_video)
    if not cap.isOpened():
        raise IOError("VideoCapture object for '{}' could not be opened".format(path_video))

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        raise IOError("Could not get frames per second for '{}'".format(path_video))

    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break

    cap.release()
    if not frames:
        raise IOError("No frames were read from '{}'".format(path_video))

    return frames, fps

def infer_model(frames, model):
    """ Run pretrained model on a consecutive list of frames    
    :params
        frames: list of consecutive video frames
        model: pretrained model
    :return    
        ball_track: list of detected ball points
        dists: list of euclidean distances between two neighbouring ball points
    """
    height = 360
    width = 640
    dists = [-1]*2
    ball_track = [(None,None)]*2
    for num in tqdm(range(2, len(frames))):
        img = cv2.resize(frames[num], (width, height))
        img_prev = cv2.resize(frames[num-1], (width, height))
        img_preprev = cv2.resize(frames[num-2], (width, height))
        imgs = np.concatenate((img, img_prev, img_preprev), axis=2)
        imgs = imgs.astype(np.float32)/255.0
        imgs = np.rollaxis(imgs, 2, 0)
        inp = np.expand_dims(imgs, axis=0)

        out = model(torch.from_numpy(inp).float().to(device))
        output = out.argmax(dim=1).detach().cpu().numpy()
        x_pred, y_pred = postprocess(output)
        ball_track.append((x_pred, y_pred))

        if ball_track[-1][0] and ball_track[-2][0]:
            dist = distance.euclidean(ball_track[-1], ball_track[-2])
        else:
            dist = -1
        dists.append(dist)  
    return ball_track, dists 

def remove_outliers(ball_track, dists, max_dist = 100):
    """ Remove outliers from model prediction    
    :params
        ball_track: list of detected ball points
        dists: list of euclidean distances between two neighbouring ball points
        max_dist: maximum distance between two neighbouring ball points
    :return
        ball_track: list of ball points
    """
    outliers = list(np.where(np.array(dists) > max_dist)[0])
    for i in outliers:
        if (dists[i+1] > max_dist) | (dists[i+1] == -1):       
            ball_track[i] = (None, None)
            outliers.remove(i)
        elif dists[i-1] == -1:
            ball_track[i-1] = (None, None)
    return ball_track  

def split_track(ball_track, max_gap=4, max_dist_gap=80, min_track=5):
    """ Split ball track into several subtracks in each of which we will perform
    ball interpolation.    
    :params
        ball_track: list of detected ball points
        max_gap: maximun number of coherent None values for interpolation  
        max_dist_gap: maximum distance at which neighboring points remain in one subtrack
        min_track: minimum number of frames in each subtrack    
    :return
        result: list of subtrack indexes    
    """
    list_det = [0 if x[0] else 1 for x in ball_track]
    groups = [(k, sum(1 for _ in g)) for k, g in groupby(list_det)]

    cursor = 0
    min_value = 0
    result = []
    for i, (k, l) in enumerate(groups):
        if (k == 1) & (i > 0) & (i < len(groups) - 1):
            dist = distance.euclidean(ball_track[cursor-1], ball_track[cursor+l])
            if (l >=max_gap) | (dist/l > max_dist_gap):
                if cursor - min_value > min_track:
                    result.append([min_value, cursor])
                    min_value = cursor + l - 1        
        cursor += l
    if len(list_det) - min_value > min_track: 
        result.append([min_value, len(list_det)]) 
    return result    

def interpolation(coords):
    """ Run ball interpolation in one subtrack    
    :params
        coords: list of ball coordinates of one subtrack    
    :return
        track: list of interpolated ball coordinates of one subtrack
    """
    def nan_helper(y):
        return np.isnan(y), lambda z: z.nonzero()[0]

    x = np.array([x[0] if x[0] is not None else np.nan for x in coords])
    y = np.array([x[1] if x[1] is not None else np.nan for x in coords])

    nons, yy = nan_helper(x)
    x[nons]= np.interp(yy(nons), yy(~nons), x[~nons])
    nans, xx = nan_helper(y)
    y[nans]= np.interp(xx(nans), xx(~nans), y[~nans])

    track = [*zip(x,y)]
    return track

def write_track(frames, ball_track, bounces, path_output_video, fps, trace=7):
    """ Write .avi file with detected ball tracks
    :params
        frames: list of original video frames
        ball_track: list of ball coordinates
        bounces: list of bounce frames
        path_output_video: path to output video
        fps: frames per second
        trace: number of frames with detected trace
    """

    height, width = frames[0].shape[:2]
    out = cv2.VideoWriter(path_output_video, cv2.VideoWriter_fourcc(*'DIVX'), 
                          fps, (width, height))
    for num in range(len(frames)):
        frame = frames[num]
        for i in range(trace):
            if (num-i > 0):
                if ball_track[num-i][0]:
                    x = int(ball_track[num-i][0])
                    y = int(ball_track[num-i][1])
                    
                    # the ball track is normally green while when the ball bounces it is red (for 2 frames)
                    color = (0,255,0) #green
                    if bounces is not None and len(bounces)>0:
                        if num in bounces or num-1 in bounces:
                            color = (0,0,255) #red
                    frame = cv2.circle(frame, (x,y), radius=0, color=color, thickness=10-i)
                else:
                    break
        out.write(frame) 
    out.release()    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    parser.add_argument('--model_path', type=str, help='path to model')
    parser.add_argument('--video_path', type=str, help='path to input video')
    parser.add_argument('--video_out_path', type=str, help='path to output video')
    parser.add_argument('--extrapolation', action='store_true', help='whether to use ball track extrapolation')
    parser.add_argument('--path_ball_track', type=str, help='path to ball track detection results (csv file)')
    parser.add_argument('--path_bounce_model', type=str, help='path to pretrained model for bounce detection')
    args = parser.parse_args()
    
    model = BallTrackerNet()
    device = 'cuda'
    model.load_state_dict(torch.load(args.model_path, weights_only=True, map_location=device))
    model = model.to(device)
    model.eval()
    
    print('Loading the video...')
    frames, fps = read_video(args.video_path)
    print('Number of frames = {}'.format(len(frames)))
    print("Ball track detection...")
    ball_track, dists = infer_model(frames, model)
    ball_track = remove_outliers(ball_track, dists)
        
    if args.extrapolation:
        print("Ball track extrapolation...")
        subtracks = split_track(ball_track)
        for r in subtracks:
            ball_subtrack = ball_track[r[0]:r[1]]
            ball_subtrack = interpolation(ball_subtrack)
            ball_track[r[0]:r[1]] = ball_subtrack
    
    #print(ball_track)
    #save the ball track in a csv file with the format: frame number, x_coord, y_coord (write -1 if coord are None)   
    if args.path_ball_track:
        with open(args.path_ball_track, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Frame Number", "X Coord", "Y Coord"])  # header row
            for i, (x, y) in enumerate(ball_track):
                x_coord = x if x is not None else -1
                y_coord = y if y is not None else -1
                writer.writerow([i+1, x_coord, y_coord])

    # bounce detection
    if args.path_bounce_model:
        print("Bounce detection...")
        bounce_detector = BounceDetector(args.path_bounce_model)
        x_ball = [x[0] for x in ball_track]
        y_ball = [x[1] for x in ball_track]
        bounces = bounce_detector.predict(x_ball, y_ball)
        write_track(frames, ball_track, bounces, args.video_out_path, fps)
    else:    
        write_track(frames, ball_track, [], args.video_out_path, fps)
    
    print("Done.")    

    



    
    
    
    
    
