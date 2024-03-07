import cv2
import os
# import torch
from extractFrames import extract_frames
from embeds import embed_text, embed_frames


def semantic_search2(text, folder_path):
    text_embedding = embed_text(text)
    videoEmbeddings = []
    similarities = []

    folder = os.listdir(folder_path)

    # For each video, embed frames and store them in an array
    # where col 0 is the file name, col 1 is the tensor of embeddings
    for video in folder:
        file = os.path.join(folder_path, video)
        frames = extract_frames(file, frame_rate=30)
        videoEmbeddings.append([video, embed_frames(frames)])

    # For each file embedding paring, find a similarity score for each frame
    for item in videoEmbeddings:
        similar = (text_embedding @ item[1].mT).squeeze(0)
        similarities.append([item[0], similar.argmax(), similar.max()])

    # separate the scores
    video_sims = [row[2] for row in similarities]

    # find the max of the scores
    most_similar = similarities[video_sims.index(max(video_sims))]

    most_similar[1] = most_similar[1].item()

    return most_similar


def main():
    folder = input('Please enter your video folder path: ')
    text = input('Please enter your search term: ')

    topFrame = semantic_search2(text, folder)

    print(f'Top Frame is in {topFrame[0]} at frame {topFrame[1]} with' +
          f'val {topFrame[2]}')

    vidcap = cv2.VideoCapture(os.path.join(folder, topFrame[0]))

    count = 0

    # Read the first frame from the video
    success, image = vidcap.read()

    # Initialize an empty list to hold the extracted frames
    frames = []

    # Loop over each frame in the video
    while success:
        # If the current frame is a multiple of the frame rate, add it to the
        # frames list
        if count % 30 == 0:
            frames.append(image)

        # Read the next frame from the video
        success, image = vidcap.read()

        # Increment the frame counter
        count += 1

    frame = frames[topFrame[1]]

    cv2.imwrite('out'+'.jpg', frame)

    return 0


if __name__ == "__main__":
    main()
