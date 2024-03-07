import streamlit as st
import os
from extractFrames import extract_frames
from embeds import embed_text, embed_frames


@st.cache_data
def extractFolder(folder_path):
    videoEmbeddings = []
    frame = []

    folder = os.listdir(folder_path)

    # For each video, embed frames and store them in an array
    # where col 0 is the file name, col 1 is the tensor of embeddings
    for video in folder:
        if '.mp4' not in video:
            continue
        file = os.path.join(folder_path, video)
        frames = extract_frames(file)
        frame.append([video, frames])
        videoEmbeddings.append([video, embed_frames(frames)])

    return (frame, videoEmbeddings)


def fetchTopK(_video_embeddings, _text_embedding, _k, _frames):
    top_matches = []

    _text_embedding /= _text_embedding.norm(dim=-1, keepdim=True)

    for vid in _video_embeddings:
        frame_features = vid[1] / vid[1].norm(dim=-1, keepdim=True)
        similar = (100 * frame_features @ _text_embedding.T).squeeze(0)
        values, indices = similar.topk(_k, 0, True, True)
        for value, index in zip(values, indices):
            top_matches.append([vid[0], index.item(), value.item()])

    topKFrames = sorted(top_matches, key=lambda x: x[2], reverse=True)[:_k]

    col1, col2, col3 = st.columns(3)

    picture_width = 200

    for i in range(len(topKFrames)):
        frameI = topKFrames[i]
        for vid in _frames:
            if vid[0] == frameI[0]:
                tempFrame = vid[1][frameI[1]]

        if i % 3 == 0:
            with col1:
                st.image(tempFrame, width=picture_width,
                         caption=(f'Frame {frameI[1]} from {frameI[0]}' +
                                  f'with value {frameI[2]}'), channels='BGR')
        elif i % 3 == 1:
            with col2:
                st.image(tempFrame, width=picture_width,
                         caption=(f'Frame {frameI[1]} from {frameI[0]}' +
                                  f'with value {frameI[2]}'), channels='BGR')
        elif i % 3 == 2:
            with col3:
                st.image(tempFrame, width=picture_width,
                         caption=(f'Frame {frameI[1]} from {frameI[0]}' +
                                  f'with value {frameI[2]}'), channels='BGR')

    return


def main():
    st.header('Image Search App')

    folder_path = '/FOLDER/PATH/HERE/'
    frames, video_embeddings = extractFolder(folder_path)

    search_term = st.text_input('Search: ')
    text_embedding = embed_text(search_term)

    st.sidebar.header('App Settings')
    k = st.sidebar.slider('Number of Search Results', min_value=1,
                          max_value=30)

    if folder_path and search_term:
        fetchTopK(video_embeddings, text_embedding, k, frames)


if __name__ == "__main__":
    main()
