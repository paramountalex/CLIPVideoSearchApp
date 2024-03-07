import streamlit as st
import os
import subprocess
import cv2
import torch
# from statistics import mean
from extractFrames import extract_frames
from embeds import embed_text, embed_frames


@st.cache_data
def extractFolderClips(folder_path, threshold):
    videoEmbeddings = []
    frame = []

    folder = os.listdir(folder_path)

    for video in folder:
        file = os.path.join(folder_path, video)

        if '.mp4' not in video:
            continue

        print(f'working on {file}')

        # use ffmpeg to get scene change frames
        # Define the FFmpeg command for scene detection
        # filters scene changes in threshold, showinfo shows frame
        # last grep strips just the frame number
        cmd_file = file
        if ' ' in cmd_file:
            cmd_file.replace(' ', '\\ ')

        cmd1 = [
            'ffmpeg',
            '-i', cmd_file,
            '-filter:v', f'select=\'gt(scene\\,{threshold}),showinfo\'',
            '-f', 'mp4', 'temp.mp4', '2>&1', '|', 'grep', 'showinfo',
            '|', 'grep', 'frame=\'[\\ 0-9.]*\'',  '-o', '|',
            'grep', '\'[0-9.]*\'', '-o',
        ]

        cmd1 = " ".join(cmd1)

        # Run FFmpeg command
        proc = subprocess.run(cmd1, shell=True, capture_output=True)

        # get the frames from the grep output and split them into a list
        scene_changes = proc.stdout.split()
        scene_changes = [int(x.decode()) for x in scene_changes]

        subprocess.run('rm temp.mp4', shell=True)

        # Create a VideoCapture object
        vidcap = cv2.VideoCapture(file)

        frame_rate = vidcap.get(cv2.CAP_PROP_FPS)

        # index all frames
        print('extracting frames...')
        frames = extract_frames(file)
        frame.append([file, frames])
        print('embedding frames...')
        videoEmbeddings.append([file, frame_rate, scene_changes,
                                embed_frames(frames)])

    return (frame, videoEmbeddings)


def fetchTopKClips(_video_embeddings, _text_embedding, _k, _frames):

    # top_matches = []

    _text_embedding /= _text_embedding.norm(dim=-1, keepdim=True)
    topClips = []

    for vid in _video_embeddings:
        # Get the embeddings of the frames and compare them to text
        frame_features = vid[3] / vid[3].norm(dim=-1, keepdim=True)
        similar = (100 * frame_features @ _text_embedding.T).squeeze(0)

        clip_averages = []

        # if unable to run ffmpeg scene detection, treat as a single clip where
        # the start time is the most similar frame
        if len(vid[2]) == 0:
            print(f'Hey, ffmpeg didn\'t run scene detection for{vid[0]}...')
            print(f'max score of video was {similar.max()}')
            topClips.append([vid[0], vid[1], False, similar.argmax().item(),
                            similar.max().item()])
        else:

            # if scenes detected, take the average similarity of every frame
            # in that shot

            # vid[1] holds fps, vid[2][0] holds first scene change
            temp_clips = similar[:round(vid[2][0]/vid[1])]
            clip_averages.append(temp_clips.mean().item())

            i = 0
            while i < (len(vid[2]) - 2):
                start = (round(vid[2][i]/vid[1]))
                end = (round(vid[2][i+1]/vid[1]))
                temp_clips = similar[start:end]
                clip_averages.append(temp_clips.mean().item())
                i += 1

            temp_clips = similar[round(vid[2][len(vid[2]) - 1]/vid[1]):]
            clip_averages.append(temp_clips.mean().item())
            clip_averages = torch.FloatTensor(clip_averages).nan_to_num(0)
            values, indices = clip_averages.topk(min(_k, len(clip_averages)),
                                                 0, True, True)

            # the top k clips and their average, or the average of every shot
            # if shot # < k
            for value, index in zip(values, indices):
                topClips.append([vid[0], vid[1], True, vid[2][index.item()],
                                 value.item()])

    # Take the top k of the shorter list of the top k from each video
    topKResults = sorted(topClips, key=lambda x: x[4], reverse=True)[:_k]

    # display the videos
    col1, col2, col3 = st.columns(3)

    for i in range(len(topKResults)):
        clipI = topKResults[i]

        # clipI[0] is filename, [3] is frame where clip starts, [1] is the
        # frame rate. This converts start frame to time

        if i % 3 == 0:
            with col1:
                st.video(data=clipI[0], start_time=round(clipI[3]/clipI[1]))

        elif i % 3 == 1:
            with col2:
                st.video(data=clipI[0], start_time=round(clipI[3]/clipI[1]))

        elif i % 3 == 2:
            with col3:
                st.video(data=clipI[0], start_time=round(clipI[3]/clipI[1]))

    return


def main():
    st.header('Video Search App')

    folder_path = ('/FOLDER/PATH/HERE/')
    threshold = 0.4
    frames, video_embeddings = extractFolderClips(folder_path, threshold)

    search_term = st.text_input('Search: ')
    text_embedding = embed_text(search_term)

    st.sidebar.header('App Settings')
    k = st.sidebar.slider('Number of Search Results', min_value=1,
                          max_value=30)

    if folder_path and search_term:
        fetchTopKClips(video_embeddings, text_embedding, k, frames)

    return


if __name__ == "__main__":
    main()
