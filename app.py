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

        from PIL import Image, ImageDraw, ImageFont
        import subprocess

        # Sidebar config
        st.sidebar.markdown("---")
        st.sidebar.subheader("Rendering Settings")
        
        resolution = st.sidebar.selectbox(
            "Video Resolution", ["360p", "480p", "720p", "1080p"], index=2,
            help="Select the target video height. Higher resolutions are sharper but take slightly longer to render."
        )
        fps_speed = st.sidebar.slider(
            "Playback Speed (FPS)", min_value=10, max_value=100, value=50, step=5,
            help="Playback speed of the exported MP4 video."
        )

        @st.cache_data(show_spinner=False, persist=True)
        def generate_video(video_idx, _data_array, res_str, fps):
            data = _data_array[video_idx] # Shape: (160, 953)
            cumulative_data = np.cumsum(data, axis=1) # Shape: (160, 953)
            
            target_h = int(res_str.replace('p', ''))
            scale = target_h / 500.0
            width, height = int(800 * scale), target_h
            # Make sure width and height are even for ffmpeg yuv420p
            width = width if width % 2 == 0 else width + 1
            height = height if height % 2 == 0 else height + 1
            
            hex_radius = 20 * scale

            cols = 16
            rows = int(np.ceil(n_neurons / cols))

            centers = []
            for i in range(n_neurons):
                r = i // cols
                c = i % cols
                x = c * hex_radius * 1.732 + (r % 2) * hex_radius * 0.866
                y = r * hex_radius * 1.5
                centers.append((x, y))

            min_x = min(c[0] for c in centers)
            max_x = max(c[0] for c in centers)
            min_y = min(c[1] for c in centers)
            max_y = max(c[1] for c in centers)
            shift_x = (width - (max_x - min_x)) / 2 - min_x
            shift_y = (height - (max_y - min_y)) / 2 - min_y

            centers = [(int(x + shift_x), int(y + shift_y)) for x, y in centers]

            polygons = []
            for cx, cy in centers:
                points = []
                for j in range(6):
                    angle = 2 * np.pi / 6 * (j + 0.5)
                    px = cx + hex_radius * np.cos(angle)
                    py = cy + hex_radius * np.sin(angle)
                    points.append((px, py))
                polygons.append(points)

            bg_color = (14, 17, 23)
            base_img = Image.new('RGB', (width, height), bg_color)
            
            font_size = max(6, int(10 * scale))
            try:
                # Optionally use a nice font if available, fallback to default
                font = ImageFont.truetype("arial.ttf", font_size)
            except IOError:
                font = ImageFont.load_default()

            vid_path = f"/tmp/vid_{video_idx}_{fps}_{res_str}_{uuid.uuid4().hex}.mp4"
            
            # Using rawvideo over a pipe makes the subprocess lightning fast compared to file writing
            ffmpeg_cmd = [
                'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
                '-s', f'{width}x{height}', '-pix_fmt', 'rgb24', '-r', str(fps),
                '-i', '-', '-c:v', 'libx264', '-preset', 'fast', '-b:v', '2M',
                '-pix_fmt', 'yuv420p', vid_path
            ]
            
            progress_bar = st.progress(0, text="Rendering fast video frames...")
            process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
            
            outline_width = max(1, int(2 * scale))
            
            for t in range(n_frames):
                if t % 50 == 0:
                    pct = int((t / n_frames) * 100)
                    progress_bar.progress(pct, text=f"Rendering fast video frames... ({pct}%)")
                    
                img = base_img.copy()
                draw = ImageDraw.Draw(img)
                
                for i in range(n_neurons):
                    val = data[i, t]
                    color = (239, 68, 68) if val == 1 else (30, 30, 30)
                    draw.polygon(polygons[i], fill=color, outline=(85, 85, 85), width=outline_width)
                    tot = cumulative_data[i, t]
                    
                    # centering text roughly
                    txt = f"Total: {tot}"
                    
                    text_x_offset = int(18 * scale)
                    text_y_offset = int(6 * scale)
                    draw.text((centers[i][0]-text_x_offset, centers[i][1]-text_y_offset), txt, fill=(255,255,255), font=font)
                    
                process.stdin.write(img.tobytes())

            process.stdin.close()
            process.wait()
            progress_bar.empty()
            
            with open(vid_path, "rb") as f:
                video_bytes = f.read()
            
            if os.path.exists(vid_path):
                os.remove(vid_path)
                
            return video_bytes

        def _draw_aggregate_frame(_data_array, frame_idx, high_data, n_vids, width=1280, height=720, scale=1.0):
            n_neurs = _data_array.shape[1]
            agg_data = np.sum(_data_array, axis=0)
            current_data = agg_data[:, frame_idx]
            
            hex_radius = 35 * scale

            cols = 16
            centers = []
            for i in range(n_neurs):
                r = i // cols
                c = i % cols
                x = c * hex_radius * 1.732 + (r % 2) * hex_radius * 0.866
                y = r * hex_radius * 1.5
                centers.append((x, y))

            min_x = min(c[0] for c in centers)
            max_x = max(c[0] for c in centers)
            min_y = min(c[1] for c in centers)
            max_y = max(c[1] for c in centers)
            shift_x = (width - (max_x - min_x)) / 2 - min_x
            shift_y = (height - (max_y - min_y)) / 2 - min_y

            centers = [(int(x + shift_x), int(y + shift_y)) for x, y in centers]

            polygons = []
            for cx, cy in centers:
                points = []
                for j in range(6):
                    angle = 2 * np.pi / 6 * (j + 0.5)
                    px = cx + hex_radius * np.cos(angle)
                    py = cy + hex_radius * np.sin(angle)
                    points.append((px, py))
                polygons.append(points)

            bg_color = (14, 17, 23)
            img = Image.new('RGB', (width, height), bg_color)
            draw = ImageDraw.Draw(img)
            
            fs1 = max(6, int(12 * scale))
            fs2 = max(8, int(13 * scale))
            try:
                font = ImageFont.truetype("arial.ttf", fs1)
                font_bold = ImageFont.truetype("arialbd.ttf", fs2)
            except IOError:
                font = font_bold = ImageFont.load_default()
                
            outline_w = max(1, int(2 * scale))
            
            for i in range(n_neurs):
                count = current_data[i]
                high_count = high_data[i]
                
                # Interpolate color between dark grey (30,30,30) and red (239,68,68) based on ratio
                ratio = count / max(1, n_vids)
                r_col = int(30 + ratio * (239 - 30))
                g_col = int(30 + ratio * (68 - 30))
                b_col = int(30 + ratio * (68 - 30))
                fill_color = (r_col, g_col, b_col)
                
                draw.polygon(polygons[i], fill=fill_color, outline=(85, 85, 85), width=outline_w)
                
                pct = (count / n_vids) * 100
                txt1 = f"High: {high_count}"
                txt2 = f"{count}/{n_vids}"
                txt3 = f"{pct:.0f}%"
                
                cx, cy = centers[i]
                offset_x1 = int(-20 * scale)
                offset_y1 = int(-15 * scale)
                offset_x2 = int(-18 * scale)
                offset_y2 = 0
                offset_x3 = int(-12 * scale)
                offset_y3 = int(15 * scale)
                
                draw.text((cx+offset_x1, cy+offset_y1), txt1, fill=(255,255,255), font=font_bold)
                draw.text((cx+offset_x2, cy+offset_y2), txt2, fill=(255,255,255), font=font)
                draw.text((cx+offset_x3, cy+offset_y3), txt3, fill=(255,255,255), font=font)
                
            return img

        @st.cache_data(show_spinner=False, persist=True)
        def plot_aggregate_frame(_data_array, frame_idx):
            high_data = np.max(np.sum(_data_array, axis=0), axis=1)
            n_vids = _data_array.shape[0]
            # Extra large for static frame explorer
            return _draw_aggregate_frame(_data_array, frame_idx, high_data, n_vids, width=1600, height=900, scale=1.5)

        @st.cache_data(show_spinner=False, persist=True)
        def generate_aggregate_video(_data_array, res_str, fps):
            n_vids, n_neurs, n_frms = _data_array.shape
            high_data = np.max(np.sum(_data_array, axis=0), axis=1)

            target_h = int(res_str.replace('p', ''))
            scale = target_h / 720.0
            width, height = int(1280 * scale), target_h
            # Make sure width and height are even for ffmpeg yuv420p
            width = width if width % 2 == 0 else width + 1
            height = height if height % 2 == 0 else height + 1

            vid_path = f"/tmp/agg_vid_{fps}_{res_str}_{uuid.uuid4().hex}.mp4"
            
            ffmpeg_cmd = [
                'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
                '-s', f'{width}x{height}', '-pix_fmt', 'rgb24', '-r', str(fps),
                '-i', '-', '-c:v', 'libx264', '-preset', 'fast', '-b:v', '4M',
                '-pix_fmt', 'yuv420p', vid_path
            ]
            
            progress_bar = st.progress(0, text="Rendering fast aggregate video frames...")
            process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
            
            for t in range(n_frms):
                if t % 50 == 0:
                    pct = int((t / n_frms) * 100)
                    progress_bar.progress(pct, text=f"Rendering fast aggregate video frames... ({pct}%)")
                    
                img = _draw_aggregate_frame(_data_array, t, high_data, n_vids, width=width, height=height, scale=scale)
                process.stdin.write(img.tobytes())
                
            process.stdin.close()
            process.wait()
            progress_bar.empty()
            
            with open(vid_path, "rb") as f:
                video_bytes = f.read()
            
            if os.path.exists(vid_path):
                os.remove(vid_path)
                
            return video_bytes

        tab1, tab2 = st.tabs(["Individual Video Playback", "Aggregate Activity Playback"])

        with tab1:
            st.subheader(f"Video {selected_video} Playback")
            if st.button("Render Individual Video", key="btn_indv", type="primary"):
                with st.spinner(f"Generating optimized playback for Video {selected_video}..."):
                    video_bytes = generate_video(selected_video, bint, resolution, fps_speed)
                st.video(video_bytes)
            
        with tab2:
            st.subheader("Aggregated Neural Firing Over All Videos")
            st.markdown("This video shows the summation of neuron firings across all 297 videos. Observe the entire clip below, and use the **Frame Explorer** to scrub frame-by-frame without skipping values.")
            
            if st.button("Render Aggregate Video", key="btn_agg", type="primary"):
                with st.spinner("Generating crisp, high-res aggregated playback (~5 seconds)..."):
                    agg_video_bytes = generate_aggregate_video(bint, resolution, fps_speed)
                st.video(agg_video_bytes)

            st.markdown("---")
            st.subheader("Frame-by-Frame Static Explorer")
            frame_slider = st.slider("Select Exact Frame", min_value=0, max_value=n_frames-1, value=0, step=1)
            
            # Using st.image since we return a PIL Image now
            img_frame = plot_aggregate_frame(bint, frame_slider)
            st.image(img_frame, use_column_width=True)

else:
    st.info("Awaiting file upload. Please select a `.mat` file from the sidebar.")

