import subprocess
import os
import cv2


def split_video_into_scenes(input_video, output_folder, threshold=0.3):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Define the FFmpeg command for scene detection
    cmd1 = [
        'ffmpeg',
        '-i', input_video,
        '-filter:v', f'select=\'gt(scene\\,{threshold}),showinfo\'',
        '-f', 'mp4', 'temp.mp4', '2>&1', '|', 'grep', 'showinfo',
        '|', 'grep', 'frame=\'[\\ 0-9.]*\'',  '-o', '|',
        'grep', '\'[0-9.]*\'', '-o',
    ]

    cmd1 = " ".join(cmd1)

    # Run FFmpeg command
    proc = subprocess.run(cmd1, shell=True, capture_output=True)

    scenes = proc.stdout.split()
    scenes = [int(x.decode()) for x in scenes]

    subprocess.run('rm temp.mp4', shell=True)
    print(scenes)

    return scenes


def embed_scenes(video_path, scenes):
    # Create a VideoCapture object
    vidcap = cv2.VideoCapture(video_path)

    frame_rate = vidcap.get(cv2.CAP_PROP_FPS)
    frame_rate = round(frame_rate)

    # Initialize the frame counter
    count = 0

    # Track which scene we're in
    scene = 0

    # Read the first frame from the video
    success, image = vidcap.read()

    # Initialize an empty list to hold the extracted frames
    frames = []
    scene_frames = []

    # Loop over each frame in the video
    while success:
        # If the current frame is a multiple of the frame rate, add it to the
        # frames list
        if count % frame_rate == 0:
            scene_frames.append(image)

        if scenes is not None:
            if count == scenes[scene]:
                frames.append(scene_frames)
                scene_frames = []
                scene += 1

        # Read the next frame from the video
        success, image = vidcap.read()

        # Increment the frame counter
        count += 1

    lens = [len(x) for x in frames]

    print(lens)

    # Return the list of extracted frames
    return frames


if __name__ == '__main__':
    # Replace with your input video file
    input_video = ('/Users/alexander.johnson/Downloads/'
                   + 'Test_Videos/BigBuckBunny.mp4')

    output_folder = 'scenes'  # Output folder to store individual scenes
    # Adjust this threshold to control scene detection sensitivity
    threshold = 0.3

    scenes = split_video_into_scenes(input_video, output_folder, threshold)
    frames = embed_scenes(input_video, scenes)
