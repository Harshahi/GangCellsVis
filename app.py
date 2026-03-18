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
import tempfile

# Configure Streamlit layout
st.set_page_config(page_title="Neural Hex Grid", layout="wide")

st.title("Hexagonal Neural Grid Visualization")
st.markdown("Observe the firing of 160 neurons over a 19-second video. Please upload your `.mat` data file containing the `bint` variable.")

uploaded_file = st.sidebar.file_uploader("Upload Data file (.mat)", type=["mat"])

@st.cache_data
def load_data(file_buffer):    
    # Write to temp file because scipy loadmat doesn't accept stream for newer mat files easily sometimes
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mat") as tmp:
        tmp.write(file_buffer.getvalue())
        tmp_path = tmp.name
        
    mat = scipy.io.loadmat(tmp_path)
    os.remove(tmp_path)
    return mat.get('bint', None)

if uploaded_file is not None:
    bint = load_data(uploaded_file)
    
    if bint is None:
        st.error("Uploaded `.mat` file does not contain a `bint` variable.")
    else:
        n_videos, n_neurons, n_frames = bint.shape

        # Sidebar
        st.sidebar.header("Configuration")
        selected_video = st.sidebar.selectbox("Select Video", options=list(range(n_videos)), index=0)

        @st.cache_data(show_spinner=False)
        def generate_video(video_idx, _data_array):
            data = _data_array[video_idx] # Shape: (160, 953)
            
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

        @st.cache_data(show_spinner=False)
        def plot_aggregate_frame(_data_array, frame_idx):
            n_vids, n_neurs, n_frms = _data_array.shape
            agg_data = np.sum(_data_array, axis=0) # Shape: (160, 953)
            current_data = agg_data[:, frame_idx]
            
            # Create high-res matplotlib figure
            fig, ax = plt.subplots(figsize=(15, 9), dpi=120)
            fig.patch.set_facecolor('#0E1117')
            ax.set_facecolor('#0E1117')
            
            cols = 16
            rows = int(np.ceil(n_neurs / cols))

            patches = []
            # Build larger hex grid (pointy-topped)
            hex_radius = 0.5 / np.sqrt(3) * 1.05 # slightly larger
            for i in range(n_neurs):
                r = i // cols
                c = i % cols
                x = c + (r % 2) * 0.5
                y = r * np.sqrt(3) / 2
                patches.append(RegularPolygon((x, y), numVertices=6, radius=hex_radius, orientation=np.pi/2))
                
                count = current_data[i]
                pct = (count / n_vids) * 100
                ax.text(x, y, f"{count}/{n_vids}\n{pct:.0f}%", ha='center', va='center', color='white', fontsize=7, fontweight='bold')

            cmap = mcolors.LinearSegmentedColormap.from_list("agg_red", ['#1E1E1E', '#EF4444'])
            
            collection = PatchCollection(patches, cmap=cmap, edgecolor='#555555', linewidth=1.5)
            collection.set_array(current_data)
            collection.set_clim(0, n_vids)
            ax.add_collection(collection)
            
            ax.set_aspect('equal')
            ax.axis('off')
            ax.autoscale_view()
            return fig

        @st.cache_data(show_spinner=False)
        def generate_aggregate_video(_data_array):
            n_vids, n_neurs, n_frms = _data_array.shape
            agg_data = np.sum(_data_array, axis=0) # Shape: (160, 953)
            
            # Create high-res matplotlib figure for video
            fig, ax = plt.subplots(figsize=(15, 9), dpi=120)
            fig.patch.set_facecolor('#0E1117')
            ax.set_facecolor('#0E1117')
            
            cols = 16
            rows = int(np.ceil(n_neurs / cols))

            patches = []
            texts = []
            
            hex_radius = 0.5 / np.sqrt(3) * 1.05 # slightly larger
            
            for i in range(n_neurs):
                r = i // cols
                c = i % cols
                x = c + (r % 2) * 0.5
                y = r * np.sqrt(3) / 2
                patches.append(RegularPolygon((x, y), numVertices=6, radius=hex_radius, orientation=np.pi/2))
                txt = ax.text(x, y, "", ha='center', va='center', color='white', fontsize=7, fontweight='bold')
                texts.append(txt)

            cmap = mcolors.LinearSegmentedColormap.from_list("agg_red", ['#1E1E1E', '#EF4444'])
            
            collection = PatchCollection(patches, cmap=cmap, edgecolor='#555555', linewidth=1.5)
            collection.set_array(agg_data[:, 0])
            collection.set_clim(0, n_vids)
            ax.add_collection(collection)
            
            ax.set_aspect('equal')
            ax.axis('off')
            ax.autoscale_view()

            def init():
                collection.set_array(agg_data[:, 0])
                for i, count in enumerate(agg_data[:, 0]):
                    pct = (count / n_vids) * 100
                    texts[i].set_text(f"{count}/{n_vids}\n{pct:.0f}%")
                return [collection] + texts

            def update(frame):
                current_data = agg_data[:, frame]
                collection.set_array(current_data)
                for i, count in enumerate(current_data):
                    pct = (count / n_vids) * 100
                    texts[i].set_text(f"{count}/{n_vids}\n{pct:.0f}%")
                return [collection] + texts

            # Generate animation (use higher bitrate for quality)
            ani = animation.FuncAnimation(fig, update, frames=n_frms, init_func=init, blit=True)
            
            vid_path = f"/tmp/agg_vid_{uuid.uuid4().hex}.mp4"
            ani.save(vid_path, fps=50, extra_args=['-vcodec', 'libx264', '-b:v', '5M'])
            plt.close(fig)
            
            with open(vid_path, "rb") as f:
                video_bytes = f.read()
            
            if os.path.exists(vid_path):
                os.remove(vid_path)
                
            return video_bytes

        tab1, tab2 = st.tabs(["Individual Video Playback", "Aggregate Activity Playback"])

        with tab1:
            st.subheader(f"Video {selected_video} Playback")
            with st.spinner(f"Generating high-performance playback for Video {selected_video} (takes ~5-10 seconds the first time)..."):
                video_bytes = generate_video(selected_video, bint)
            st.video(video_bytes)
            
        with tab2:
            st.subheader("Aggregated Neural Firing Over All Videos")
            st.markdown("This video shows the summation of neuron firings across all 297 videos. Observe the entire clip below, and use the **Frame Explorer** to scrub frame-by-frame without skipping values.")
            
            with st.spinner("Generating crisp, high-res aggregated playback (takes ~15-30 seconds the first time)..."):
                agg_video_bytes = generate_aggregate_video(bint)
            st.video(agg_video_bytes)

            st.markdown("---")
            st.subheader("Frame-by-Frame Explorer")
            frame_slider = st.slider("Select Exact Frame", min_value=0, max_value=n_frames-1, value=0, step=1)
            
            with st.spinner(f"Rendering frame {frame_slider}..."):
                fig_frame = plot_aggregate_frame(bint, frame_slider)
                st.pyplot(fig_frame)

else:
    st.info("Awaiting file upload. Please select a `.mat` file from the sidebar.")

