import cv2
import os
from tqdm import tqdm
NUM_OF_EPS = 9

directory = 'vidsPics'
os.chdir(directory)

for ep in tqdm(range(17, 23)):
    video = cv2.VideoCapture(f"../vids/Nothing/ep{ep}.mp4")
    # Get the total number of frames in the video
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Set the frame extraction interval
    interval = 2

    # Loop over the frames
    for i in range(0, total_frames, int(interval * video.get(cv2.CAP_PROP_FPS))):
        video.set(cv2.CAP_PROP_POS_FRAMES, i)

        # Read the frame from the video
        ret, frame = video.read()

        # Save the frame as an image
        if frame is not None and frame.size:
            cv2.imwrite(f"ep{ep}frame{i}.jpg", frame)
    # Release the video capture
    video.release()
