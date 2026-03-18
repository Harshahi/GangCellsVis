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

        import json
        import streamlit.components.v1 as components
        import pandas as pd

        # Sidebar config
        st.sidebar.markdown("---")
        st.sidebar.subheader("Rendering Settings")
        
        resolution = st.sidebar.selectbox(
            "Video Resolution", ["360p", "480p", "720p", "1080p"], index=2,
            help="Select the target video height. Higher resolutions are sharper but take slightly longer to render."
        )
        fps_speed = st.sidebar.slider(
            "Playback Speed (FPS)", min_value=10, max_value=100, value=50, step=5,
            help="Playback speed of the exported visualization."
        )

        def render_html5_viewer(data_matrix, cumulative_data_matrix, high_data, fps, mode="individual", scale=1.0, n_vids=297):
            width = int(800 * scale)
            height = int(500 * scale)
            if mode == "aggregate":
                width = int(1280 * scale)
                height = int(720 * scale)
                hex_radius = 35 * scale
            else:
                hex_radius = 20 * scale

            cols = 16
            n_neurons = data_matrix.shape[0]
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

            data_json = json.dumps(data_matrix.tolist())
            cumulative_json = json.dumps(cumulative_data_matrix.tolist()) if cumulative_data_matrix is not None else "null"
            high_json = json.dumps(high_data.tolist()) if high_data is not None else "null"
            polygons_json = json.dumps(polygons)

            fs_main = max(5, int(8*scale)) if mode == "individual" else max(5, int(9*scale))
            fs_bold = max(6, int(10*scale))
            line_w = max(1, int(2*scale))

            html_code = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    body {{ background-color: #0E1117; color: white; font-family: sans-serif; margin: 0; padding: 10px; }}
                    #controls {{ margin-bottom: 15px; display: flex; align-items: center; gap: 15px; background: #1E1E1E; padding: 10px; border-radius: 8px; }}
                    button {{ background: #EF4444; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; font-weight: bold; }}
                    button:hover {{ background: #DC2626; }}
                    input[type=range] {{ flex-grow: 1; accent-color: #EF4444; }}
                    canvas {{ border: 1px solid #333; border-radius: 8px; }}
                    #frameInput {{ width: 60px; background: #333; color: white; border: 1px solid #555; padding: 4px; border-radius: 4px; text-align: center; }}
                    #frameInput:focus {{ outline: none; border-color: #EF4444; }}
                </style>
            </head>
            <body>
                <div id="controls">
                    <button id="playBtn">Play</button>
                    <button id="pauseBtn">Pause</button>
                    <input type="range" id="frameSlider" min="0" max="{data_matrix.shape[1]-1}" value="0">
                    <span>Frame:</span>
                    <input type="number" id="frameInput" min="0" max="{data_matrix.shape[1]-1}" value="0">
                </div>
                <canvas id="hexCanvas" width="{width}" height="{height}"></canvas>

                <script>
                    const data = {data_json};
                    const cumulativeData = {cumulative_json};
                    const highData = {high_json};
                    const polygons = {polygons_json};
                    const mode = "{mode}";
                    const n_vids = {n_vids};
                    
                    const canvas = document.getElementById('hexCanvas');
                    const ctx = canvas.getContext('2d');
                    const frameSlider = document.getElementById('frameSlider');
                    const frameInput = document.getElementById('frameInput');
                    
                    let currentFrame = 0;
                    const totalFrames = {data_matrix.shape[1]};
                    let isPlaying = false;
                    let lastTime = 0;
                    const fpsInterval = 1000 / {fps};
                    
                    ctx.textAlign = "center";
                    ctx.textBaseline = "middle";

                    function drawPolygon(points, fillStyle, strokeStyle, lineWidth) {{
                        ctx.beginPath();
                        ctx.moveTo(points[0][0], points[0][1]);
                        for(let i=1; i<points.length; i++) {{
                            ctx.lineTo(points[i][0], points[i][1]);
                        }}
                        ctx.closePath();
                        ctx.fillStyle = fillStyle;
                        ctx.fill();
                        ctx.lineWidth = lineWidth;
                        ctx.strokeStyle = strokeStyle;
                        ctx.stroke();
                    }}

                    function renderFrame() {{
                        ctx.fillStyle = '#0E1117';
                        ctx.fillRect(0, 0, canvas.width, canvas.height);
                        
                        for(let i=0; i<polygons.length; i++) {{
                            const val = data[i][currentFrame];
                            const pts = polygons[i];
                            
                            let cx = 0, cy = 0;
                            for(let p of pts) {{ cx += p[0]; cy += p[1]; }}
                            cx /= pts.length;
                            cy /= pts.length;

                            if (mode === "individual") {{
                                const color = val === 1 ? 'rgb(239, 68, 68)' : 'rgb(30, 30, 30)';
                                drawPolygon(pts, color, 'rgb(85, 85, 85)', {line_w});
                                
                                const tot = cumulativeData[i][currentFrame];
                                ctx.fillStyle = 'white';
                                ctx.font = "{fs_main}px Arial";
                                ctx.fillText(tot.toString(), cx, cy);
                            }} else {{
                                const ratio = val / Math.max(1, n_vids);
                                const r = Math.floor(30 + ratio * (239 - 30));
                                const g = Math.floor(30 + ratio * (68 - 30));
                                const b = Math.floor(30 + ratio * (68 - 30));
                                const color = `rgb(${{r}}, ${{g}}, ${{b}})`;
                                
                                drawPolygon(pts, color, 'rgb(85, 85, 85)', {line_w});
                                
                                const high = highData[i];
                                const pct = (val / n_vids * 100).toFixed(0) + '%';
                                const countStr = val + "/" + n_vids;
                                
                                ctx.fillStyle = 'white';
                                ctx.font = "bold {fs_bold}px Arial";
                                ctx.fillText("H: " + high, cx, cy - {int(9*scale)});
                                
                                ctx.font = "{fs_main}px Arial";
                                ctx.fillText(countStr, cx, cy + {int(1*scale)});
                                ctx.fillText(pct, cx, cy + {int(10*scale)});
                            }}
                        }}
                    }}

                    function loop(time) {{
                        if (!isPlaying) return;
                        
                        const elapsed = time - lastTime;
                        if (elapsed > fpsInterval) {{
                            lastTime = time - (elapsed % fpsInterval);
                            currentFrame++;
                            if (currentFrame >= totalFrames) {{
                                currentFrame = 0;
                                isPlaying = false; 
                                renderFrame();
                                return;
                            }}
                            frameSlider.value = currentFrame;
                            frameInput.value = currentFrame;
                            renderFrame();
                        }}
                        requestAnimationFrame(loop);
                    }}

                    document.getElementById('playBtn').addEventListener('click', () => {{
                        if (!isPlaying) {{
                            isPlaying = true;
                            if (currentFrame >= totalFrames - 1) currentFrame = 0;
                            lastTime = performance.now();
                            requestAnimationFrame(loop);
                        }}
                    }});
                    
                    document.getElementById('pauseBtn').addEventListener('click', () => {{
                        isPlaying = false;
                    }});

                    frameSlider.addEventListener('input', (e) => {{
                        currentFrame = parseInt(e.target.value);
                        frameInput.value = currentFrame;
                        renderFrame();
                    }});
                    
                    frameInput.addEventListener('change', (e) => {{
                        let val = parseInt(e.target.value);
                        if (isNaN(val)) val = 0;
                        if (val < 0) val = 0;
                        if (val >= totalFrames) val = totalFrames - 1;
                        
                        currentFrame = val;
                        e.target.value = val;
                        frameSlider.value = val;
                        renderFrame();
                    }});
                    
                    renderFrame();
                </script>
            </body>
            </html>
            """
            components.html(html_code, height=height + 80)

        tab1, tab2 = st.tabs(["Individual Video Playback", "Aggregate Activity Playback"])

        with tab1:
            st.subheader(f"Video {selected_video} Playback")
            st.markdown("The number on each neuron represents the **total accumulative firings** until the current frame.")
            
            target_h = int(resolution.replace('p', ''))
            scale_indv = target_h / 500.0
            
            data_indv = bint[selected_video]
            cumulative_indv = np.cumsum(data_indv, axis=1)
            
            render_html5_viewer(data_indv, cumulative_indv, None, fps_speed, mode="individual", scale=scale_indv)
            
        with tab2:
            st.subheader("Aggregated Neural Firing HTML5 Canvas")
            st.markdown("This canvas maps the summation of neuron firings across all 297 videos instantly directly in your browser. Use the native Play controls below.")
            
            target_h_agg = int(resolution.replace('p', ''))
            scale_agg = target_h_agg / 720.0
            
            n_vids = bint.shape[0]
            agg_data = np.sum(bint, axis=0)
            high_data = np.max(agg_data, axis=1)
            
            render_html5_viewer(agg_data, None, high_data, fps_speed, mode="aggregate", scale=scale_agg, n_vids=n_vids)

            st.markdown("---")
            st.subheader("Maximum Concurrent Firings per Neuron (H)")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                sort_order = st.selectbox(
                    "Sort Order", 
                    ["Neuron Index (Default)", "Descending (High to Low)", "Ascending (Low to High)"],
                    index=0
                )
            
            chart_data = pd.DataFrame(
                high_data,
                columns=["High Firings (H)"]
            )
            chart_data.index.name = "Neuron Index"
            
            if sort_order == "Descending (High to Low)":
                chart_data = chart_data.sort_values(by="High Firings (H)", ascending=False)
            elif sort_order == "Ascending (Low to High)":
                chart_data = chart_data.sort_values(by="High Firings (H)", ascending=True)
                
            st.bar_chart(chart_data, y="High Firings (H)", use_container_width=True)
            
            csv = chart_data.to_csv().encode('utf-8')
            with col2:
                st.markdown("<br>", unsafe_allow_html=True) # alignment spacing
                st.download_button(
                    label="Download data as CSV",
                    data=csv,
                    file_name='neuron_high_firings.csv',
                    mime='text/csv',
                )

else:
    st.info("Awaiting file upload. Please select a `.mat` file from the sidebar.")

