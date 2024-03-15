import streamlit as st
import os
import torch
import clip
import subprocess
import cv2
import threading
import lancedb
from lancedb.pydantic import Vector, LanceModel
from embeds import embed_text, embed_frames
from extractFrames import extract_all_frames
# from streamlit.runtime.scriptrunner import add_script_run_ctx


# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)


# Define the database schema
class Frame(LanceModel):
    vector: Vector(512)
    filename: str
    timestamp: int
    shot: int


# Create or connect to the database
@st.cache_resource
def connect_db():
    db = lancedb.connect("./data/db")
    table = db.create_table("videos", schema=Frame,
                            mode="overwrite")
    return table


table = connect_db()


def search_videos(query):

    print(f'searching for videos with {query}...')
    # Embed the query
    text_embedding = embed_text(query)

    text_embedding = text_embedding.detach().cpu().numpy()[0].tolist()
    # Get the video embeddings from the database
    results = table.search(text_embedding)\
                   .metric("cosine")\
                   .limit(500)\
                   .to_list()
    return results


def get_shot(frame, scene_changes):
    for i, change in enumerate(scene_changes):
        if frame < change:
            return i
    return len(scene_changes)


def embed_video(video_path, threshold=0.3):
    # use ffmpeg to get scene change frames
    # Define the FFmpeg command for scene detection
    # filters scene changes in threshold, showinfo shows frame
    # last grep strips just the frame number
    cmd_file = video_path
    if ' ' in cmd_file:
        cmd_file.replace(' ', '\\ ')

    print(f'working on {video_path}')
    pid = threading.get_native_id()

    cmd1 = [
        'ffmpeg',
        '-i', cmd_file,
        '-filter:v', f'select=\'gt(scene\\,{threshold}),showinfo\'',
        '-f', 'mp4', f'temp_{pid}.mp4', '2>&1', '|', 'grep', 'showinfo',
        '|', 'grep', 'frame=\'[\\ 0-9.]*\'',  '-o', '|',
        'grep', '\'[0-9.]*\'', '-o',
    ]
    cmd1 = " ".join(cmd1)
    # Run FFmpeg command
    proc = subprocess.run(cmd1, shell=True, capture_output=True)

    print("scene split done for ", video_path)
    # get the frames from the grep output and split them into a list
    scene_changes = proc.stdout.split()
    scene_changes = [int(x.decode()) for x in scene_changes]
    subprocess.run(f'rm temp_{pid}.mp4', shell=True)

    # Create a VideoCapture object
    vidcap = cv2.VideoCapture(video_path)

    frame_rate = vidcap.get(cv2.CAP_PROP_FPS)

    vidcap.release()

    print(f'extracting frames for {video_path}...')

    # index all frames
    frames = extract_all_frames(video_path)
    print(f'embedding frames for {video_path}...')
    embeds = embed_frames(frames)
    embeds = embeds.cpu().numpy().tolist()
    frame_list = []
    for i, embedding in enumerate(embeds):
        frame = Frame(vector=embedding[0], filename=video_path,
                      timestamp=round(i/frame_rate),
                      shot=get_shot(i, scene_changes))
        frame_list.append(frame)
    print(f'adding {len(frame_list)} frames to the database...')
    table.add(frame_list)

    print(f'finished working on {video_path}')

    return


def build_upload_tab(tab1):
    with tab1:
        st.write("Upload Videos to the Database")
        folder_path = st.text_input("Enter the video folder path")

        if st.button("Upload"):

            folder = os.listdir(folder_path)
            bar = st.progress(0)
            # threads = []
            for video in folder:
                if '.mp4' not in video:
                    continue
                file = os.path.join(folder_path, video)
                bar.progress((folder.index(video) + 1)/len(folder))
                embed_video(file)
                # thread = threading.Thread(target=embed_video,
                #                          args=[file])
                # threads.append(thread)
                # add_script_run_ctx(thread)
                # thread.start()
                # print(f'started thread for {file}')

            # while threads:
                # threads = [t for t in threads if t.is_alive()]
                # bar.progress((len(folder) - len(threads))/len(folder))

            bar.empty()

    return


def build_search_tab(tab2):
    with tab2:
        st.write("Search for Videos")
        query = st.text_input("Enter a search query")
        if query:
            results = search_videos(query)
            # Display the results
            shown = []
            col1, col2, col3 = st.columns(3)
            for result in results:
                if ([result['filename'], result['shot']] not in shown
                   and len(shown) < 15):
                    if len(shown) % 3 == 0:
                        with col1:
                            st.video(result['filename'],
                                     start_time=result['timestamp'])
                            shown.append([result['filename'], result['shot']])
                    elif len(shown) % 3 == 1:
                        with col2:
                            st.video(result['filename'],
                                     start_time=result['timestamp'])
                            shown.append([result['filename'], result['shot']])
                    elif len(shown) % 3 == 2:
                        with col3:
                            st.video(result['filename'],
                                     start_time=result['timestamp'])
                            shown.append([result['filename'], result['shot']])


def build_data_tab(tab3):
    with tab3:
        st.write("Database")
        data = table.to_pandas()
        st.write(data)


def main():
    st.title("Video Search App")
    st.write("Welcome to the Video Search App." +
             " You can search for videos using text queries.")
    tab1, tab2, tab3 = st.tabs(["Upload", "Search", "Database"])

    build_upload_tab(tab1)
    build_search_tab(tab2)
    build_data_tab(tab3)

    return


if __name__ == "__main__":
    main()
