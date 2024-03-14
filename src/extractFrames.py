# Import the OpenCV library
import cv2


# Define a function to extract frames from a video
def extract_frames(video_path):

    # Create a VideoCapture object
    vidcap = cv2.VideoCapture(video_path)

    frame_rate = vidcap.get(cv2.CAP_PROP_FPS)
    frame_rate = round(frame_rate)

    # Initialize the frame counter
    count = 0

    # Read the first frame from the video
    success, image = vidcap.read()

    # Initialize an empty list to hold the extracted frames
    frames = []

    # Loop over each frame in the video
    while success:
        # If the current frame is at a .5 second interval add it to the
        # frames list
        if count % (frame_rate // 2) == 0:
            frames.append(image)

        # Read the next frame from the video
        success, image = vidcap.read()

        # Increment the frame counter
        count += 1

    vidcap.release()

    # Return the list of extracted frames
    return frames


# Define a function to extract frames from a video
def extract_all_frames(video_path):

    # Create a VideoCapture object
    vidcap = cv2.VideoCapture(video_path)

    # Read the first frame from the video
    success, image = vidcap.read()

    # Initialize an empty list to hold the extracted frames
    frames = []

    # Loop over each frame in the video
    while success:
        frames.append(image)

        # Read the next frame from the video
        success, image = vidcap.read()

    vidcap.release()

    # Return the list of extracted frames
    return frames
