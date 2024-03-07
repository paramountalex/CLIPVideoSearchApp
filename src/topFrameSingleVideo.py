import cv2
from extractFrames import extract_frames
from embeds import embed_text, embed_frames


def semantic_search(text, video_path):
    # Extract frames from the video
    frames = extract_frames(video_path)

    # Embed the text and the frames
    text_embedding = embed_text(text)
    frame_embeddings = embed_frames(frames)

    # Compute the similarity between the text and each frame
    similarities = (text_embedding @ frame_embeddings.mT).squeeze(0)

    # Return the frame with the highest similarity
    most_similar_frame_index = similarities.argmax().item()

    return frames[most_similar_frame_index]


def main():
    video = input('Please enter your video path: ')
    text = input('Please enter your search term: ')

    frame = semantic_search(text, video)

    cv2.imwrite('out'+'.jpg', frame)

    return 0


if __name__ == "__main__":
    main()
