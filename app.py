import streamlit as st
import numpy as np
import scipy.io
import os
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from matplotlib.collections import PatchCollection
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import uuid

# Configure Streamlit layout
st.set_page_config(page_title="Neural Hex Grid", layout="wide")

st.title("Hexagonal Neural Grid Visualization")
st.markdown("Observe the firing of 160 neurons over a 19-second video. Select a video from the sidebar and use the video player controls to play, pause, or scrub.")

@st.cache_data
def load_data():
    mat = scipy.io.loadmat("bint_fishmovie32_100.mat")
    return mat['bint']

bint = load_data()
n_videos, n_neurons, n_frames = bint.shape

# Sidebar
st.sidebar.header("Configuration")
selected_video = st.sidebar.selectbox("Select Video", options=list(range(n_videos)), index=0)

@st.cache_data(show_spinner=False)
def generate_video(video_idx):
    data = bint[video_idx] # Shape: (160, 953)
    
    # Create matplotlib figure
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('#0E1117')  # Match Streamlit dark theme
    ax.set_facecolor('#0E1117')
    
    cols = 16
    rows = int(np.ceil(n_neurons / cols))

    patches = []
    # Build hex grid (pointy-topped)
    for i in range(n_neurons):
        r = i // cols
        c = i % cols
        x = c + (r % 2) * 0.5
        y = r * np.sqrt(3) / 2
        patches.append(RegularPolygon((x, y), numVertices=6, radius=0.5/np.sqrt(3), orientation=np.pi/2))

    # Dark grey for resting (0), bright red for firing (1)
    cmap = mcolors.ListedColormap(['#1E1E1E', '#EF4444'])
    
    collection = PatchCollection(patches, cmap=cmap, edgecolor='#333333', linewidth=1)
    collection.set_array(data[:, 0])
    collection.set_clim(0, 1)
    ax.add_collection(collection)
    
    ax.set_aspect('equal')
    ax.axis('off')
    ax.autoscale_view()

    def init():
        collection.set_array(data[:, 0])
        return collection,

    def update(frame):
        collection.set_array(data[:, frame])
        return collection,

    # Generate animation
    ani = animation.FuncAnimation(fig, update, frames=n_frames, init_func=init, blit=True)
    
    # Save to a temporary file, then read bytes
    vid_path = f"/tmp/vid_{video_idx}_{uuid.uuid4().hex}.mp4"
    ani.save(vid_path, fps=50, extra_args=['-vcodec', 'libx264'])
    plt.close(fig)
    
    with open(vid_path, "rb") as f:
        video_bytes = f.read()
    
    # Cleanup temp file
    if os.path.exists(vid_path):
        os.remove(vid_path)
        
    return video_bytes

st.subheader(f"Video {selected_video} Playback")

with st.spinner(f"Generating high-performance playback for Video {selected_video} (takes ~5-10 seconds the first time)..."):
    video_bytes = generate_video(selected_video)

# Display the video using Streamlit's native player
st.video(video_bytes)

