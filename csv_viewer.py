
# lunch with: python -m streamlit run .\csv_viewer.py

from matplotlib.pyplot import plot
import streamlit as st
from streamlit_modal import Modal
from streamlit_js_eval import streamlit_js_eval
import pandas as pd
import plotly.graph_objects as go
import json
import os
import numpy as np
from numpy import *
from stl import mesh # package numpy-stl
if "csv_df" not in st.session_state:
    st.session_state.csv_df = pd.DataFrame()


######  Configuration  ######
show_3d_hopper_path = False
config_path = "config.json"





config = { }

def create_default_trace2d():
    default_trace2d =   {
                        "legend": "",
                        "line": "solid",
                        "x_col_in_csv": st.session_state.csv_df.columns.tolist()[0],
                        "y_col_in_csv": st.session_state.csv_df.columns.tolist()[1],

                        "title": "2D Plot",
                        "color": '#FF0000', # Red color
                        "width": 4,
                        }
    return default_trace2d

def create_default_plot2d():
    default_plot2d =    {
                        "title": "2D Plot",
                        "xaxis_title": "Time [s]",
                        "yaxis_title": "",
                        "autoscale": True,
                        "x_min": 0,
                        "x_max": 1,
                        "y_min": 0,
                        "y_max": 1,
                        "showlegend": True,
                        "Trace 1": create_default_trace2d()
                        }
    return default_plot2d

def load_config():
    """Load the configuration from the config.json file."""
    global config
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        config = {}

def save_config():
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

def title(str):
    """Create a title for the application."""
    st.markdown(f"<h1 style='text-align: left;'>{str}</h1>", unsafe_allow_html=True)

def top_button_bar():
    """Create the top button bar for file upload."""
    
    global config

    col1, col2, _, col3 = st.columns([100, 100, 2, 50])

    csv_file = col1.file_uploader("Load CSV", type=["csv"])
    if csv_file:
        st.session_state.csv_df = pd.read_csv(csv_file)
        col1.success("CSV file loaded successfully.")

    config_file = col2.file_uploader("Load config", type=["json"])
    if config_file:
        with open(config_file, "r") as f:
            config = json.load(f)
            col2.success("Configuration loaded from {}".format(config_file.name))

    col3.markdown("<div style='height:44px'></div>", unsafe_allow_html=True)
    col3.download_button(
        label="Download config",
        data=json.dumps(config, indent=4),
        file_name="config.json",
        mime="application/json"
    )

def convert_expression_into_values(expression, df):
    for var in df.columns:
        expression = expression.replace(var, f"df['{var}']")
    return eval(expression)

def csv_variables_table():
    """Display the CSV variables in a table and alow creation of new variables via function of other variables."""
    if st.session_state.csv_df.empty:
        return
    if "variables_df" not in st.session_state:
        st.session_state.variables_df = pd.DataFrame({"variables": ["example"], "expressions":[f"{st.session_state.csv_df.columns[1]} + {st.session_state.csv_df.columns[2]}"]})
    if "variables_df_edit" not in st.session_state:
        st.session_state.variables_df_edit = st.session_state.variables_df.copy()
    if "filtered_df" not in st.session_state:
        st.session_state.filtered_df = pd.DataFrame({"variables": ["example"], "type":[f"mobile mean"]})
    if "filtered_df_edit" not in st.session_state:
        st.session_state.filtered_df_edit = st.session_state.filtered_df.copy()
    
    col_csv, col_var = st.columns([1, 1])

    col_var.markdown("### Create new variables from existing CSV columns")
    variables_df_edit = col_var.data_editor(st.session_state.variables_df,num_rows="dynamic", height= "stretch", key="variables_df_editor")
    filtered_df_edit = col_var.data_editor(st.session_state.filtered_df,num_rows="dynamic", height= "stretch", key="filtered_df_editor")
    variables_df_edit.fillna("", inplace=True)
    st.session_state.variables_df = variables_df_edit
    for i in range(len(st.session_state.variables_df["variables"])):
        if st.session_state.variables_df["variables"][i] != "" and st.session_state.variables_df["expressions"][i] != "":
            try:
                st.session_state.csv_df[st.session_state.variables_df["variables"][i]] = convert_expression_into_values(st.session_state.variables_df["expressions"][i], st.session_state.csv_df)
            except Exception as e:
                st.warning(f"Error in expression for variable '{st.session_state.variables_df['variables'][i]}': {e}")

    col_csv.markdown("### CSV Columns")
    col_csv.dataframe(st.session_state.csv_df.head(20), height= "auto")
 
def create_cylinder(center_x, center_y, center_z, radius=0.2, height=1.0, resolution=20):
    """Generate the coordinates for a 3D cylinder mesh."""
    theta = np.linspace(0, 2 * np.pi, resolution)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z_bottom = np.zeros_like(theta)
    z_top = np.ones_like(theta) * height
    x_full = np.concatenate([x, x])
    y_full = np.concatenate([y, y])
    z_full = np.concatenate([z_bottom, z_top])
    return x_full + center_x, y_full + center_y, z_full + center_z

def quaternion_multiply(q1, q2):
    """Multiply two quaternions."""
    if len(q1) == 3: 
        q1 = np.concatenate(([0], q1))
    if len(q2) == 3:
        q2 = np.concatenate(([0], q2))
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def quaternion_rotation(q, tab_p):
    """Rotate points using quaternion."""
    q = np.array(q)
    p = np.array(tab_p)
    q_conjugate = np.concatenate((q[:1], -q[1:]))
    p_rotated = [quaternion_multiply(quaternion_multiply(q, p), q_conjugate)[1:] for p in tab_p]
    return p_rotated

def anim_3d():
    """Create the 3D animation along the trajectory."""
     # Check if the CSV file is loaded
    if st.session_state.csv_df.empty:
        return

    # Check if the CSV contains the required columns
    if not {"time", "pos_x", "pos_y", "pos_z", "quat_r", "quat_x", "quat_y", "quat_z"}.issubset(st.session_state.csv_df.columns):
        st.info("The CSV file must contain the columns: pos_x, pos_y, pos_z to plot the STL animation.")
        return

    # Load STL file
    stl_path = "hopper.stl"  # Replace this path with your actual STL file path
    if not os.path.exists(stl_path):
        st.warning("STL file not found. Please check the path.")
        return

    your_mesh = mesh.Mesh.from_file(stl_path)
    vertices = your_mesh.vectors.reshape(-1, 3)*0.001  # Scale the mesh
    faces = np.arange(len(vertices)).reshape(-1, 3)

    # Create initial position
    pos = st.session_state.csv_df.iloc[0][["pos_x", "pos_y", "pos_z"]].to_numpy()
    moved_vertices = vertices + pos

    # Compute ranges and duration time
    x_range = [st.session_state.csv_df["pos_x"].min(), st.session_state.csv_df["pos_x"].max()]
    y_range = [st.session_state.csv_df["pos_y"].min(), st.session_state.csv_df["pos_y"].max()]
    z_range = [st.session_state.csv_df["pos_z"].min(), st.session_state.csv_df["pos_z"].max()]
    iso_range = [min(x_range[0], y_range[0], z_range[0]), max(x_range[1], y_range[1], z_range[1])]
    len_range = iso_range[-1] - iso_range[0]
    duration_time = (st.session_state.csv_df["time"].iloc[-1]-st.session_state.csv_df["time"][0])*1000/len(st.session_state.csv_df["time"])

    # checkbox for camera flow
    camera_follow = st.checkbox("Camera follow hopper", value=False, key="camera_follow")
    
    # Create frames for the animation
    frames = []
    for i, (_, row) in enumerate(st.session_state.csv_df.iterrows()):
        offset = row[["pos_x", "pos_y", "pos_z"]].to_numpy()
        quat = row[["quat_r", "quat_x", "quat_y", "quat_z"]].to_numpy()

        moved = quaternion_rotation(quat, vertices) + offset

        camera = dict(
            eye=dict(x=0.1, y=0.1, z=0.1),
            center=dict(x=0, y=0, z=0),
            up=dict(x=0, y=0, z=1)
        )
        if camera_follow:
            camera = dict(
                eye=dict(x=offset[0]/len_range+0.05, y=offset[1]/len_range+0.05, z=offset[2]/len_range+0.025),
                center=dict(x=offset[0]/len_range, y=offset[1]/len_range, z=(offset[2]-1)/len_range),
                up=dict(x=0, y=0, z=1)
            )

        frames.append(go.Frame(
            name=str(i),
            data=[
                go.Mesh3d(
                    x=moved[:, 0],
                    y=moved[:, 1],
                    z=moved[:, 2],
                    i=faces[:, 0],
                    j=faces[:, 1],
                    k=faces[:, 2],
                    color='gray',
                    opacity=0.5 )],
            layout=go.Layout( scene=dict(camera=camera) )
        ))


    # Slider steps
    slider_steps = [
        dict(
            method="animate",
            args=[[str(i)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}],
            label=str(i)
        )
        for i in range(len(frames))
    ]

    
    # Create the Plotly figure with initial data
    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=moved_vertices[:, 0],
                y=moved_vertices[:, 1],
                z=moved_vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color='gray',
                opacity=0.5,
                name="Hopper"
            ),
            go.Scatter3d(
                x=st.session_state.csv_df["pos_x"],
                y=st.session_state.csv_df["pos_y"],
                z=st.session_state.csv_df["pos_z"],
                mode='lines',
                line=dict(color='blue', width=4),
                name="Trajectory"
            )
        ],
        layout=go.Layout(
            scene=dict(
                xaxis_range=iso_range,
                yaxis_range=iso_range,
                zaxis_range=iso_range,
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                bgcolor='white',
                aspectmode='cube',
            ),
            margin=dict(l=0, r=0, b=0, t=0),
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[None, {
                                "frame": {"duration": duration_time, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                                "repeat": True
                            }]
                        ),
                        dict(
                            label="Pause",
                            method="animate",
                            args=[[None], {
                                "mode": "immediate",
                                "frame": {"duration": 0, "redraw": False},
                                "transition": {"duration": 0},
                                "repeat": True
                            }]
                        )
                    ]
                )
            ],
            sliders=[
                dict(
                    active=0,
                    currentvalue={"prefix": "Frame: "},
                    pad={"t": 50},
                    steps=slider_steps
                )
            ]
        ),
        frames=frames
    )


    _, col_plot3d, _ = st.columns([1, 1, 1])
    with col_plot3d:
        st.plotly_chart(fig, use_container_width=True, key="anim3d", height=900)

def plot_2d_window(container, n_plot):
    """Create the 2D plot with option and delete buttons."""

    global config
    plot = config["plot2d"][n_plot]

    # Check keys in plot
    if not set(create_default_plot2d().keys()).issubset(set(plot.keys())):
        plot = create_default_plot2d()

    # Create the buttons columns
    _, col_title, col_option, col_delete, _ = container.columns([1, 6, 1, 1, 1])
    
    # Plot title
    col_title.markdown(f"<h3 style='text-align: left;'>{plot['title']}</h3>", unsafe_allow_html=True)

    # Plot options 
    modal = Modal("Plot Options", key=f"modal_{n_plot}")
    if col_option.button("⚙️", key=f"options_{n_plot}"):
        modal.open()

    # Delete the plot
    if col_delete.button("❌", key=f"delete_{n_plot}"):
        config["plot2d"].pop(n_plot)
        save_config()
        st.rerun()
        return

    # Modal for plot options 
    if modal.is_open():
        with modal.container():
            st.markdown("<div style='width:744px; display:inline-block; visibility:hidden'></div>", unsafe_allow_html=True)
            plot["title"] = st.text_input("Title", plot["title"])
            plot["autoscale"] = st.checkbox("Autoscale", plot["autoscale"])
            if not plot["autoscale"]:
                plot["x_min"] = st.number_input("X min", value=plot["x_min"])
                plot["x_max"] = st.number_input("X max", value=plot["x_max"])
                plot["y_min"] = st.number_input("Y min", value=plot["y_min"])
                plot["y_max"] = st.number_input("Y max", value=plot["y_max"])
            plot["xaxis_title"] = st.text_input("Title X Axis", plot["xaxis_title"])
            plot["yaxis_title"] = st.text_input("Title Y Axis", plot["yaxis_title"])
            plot["showlegend"] = st.checkbox("Show Legend", plot["showlegend"])


            st.markdown("### Traces Options")
            c1, c2, c3 = st.columns([3, 1, 1])

            traces_list = [s for s in plot.keys() if s.strip().startswith("Trace")]
            traces_list = sorted(traces_list, key=lambda x: int(x.split()[1]))
            trace = c1.selectbox("Trace selection", traces_list, index=0)
            
            if c2.button("Add New Trace", key=f"add_trace_{n_plot}"):
                nb_trace = len(traces_list)
                plot[f"Trace {nb_trace + 1}"] = create_default_trace2d()
                save_config()
                st.rerun()
            
            if len(traces_list) > 1 and c3.button("Delete This Trace", key=f"delete_trace_{n_plot}"):
                plot.pop(trace)
                # Recréer un dictionnaire avec traces renumérotées
                traces = sorted([k for k in plot.keys() if k.startswith("Trace")], key=lambda x: int(x.split()[1]))
                new_plot_traces = {}
                for idx, key in enumerate(traces, start=1):
                    new_plot_traces[f"Trace {idx}"] = plot[key]
                # Delete old traces
                for key in list(plot.keys()):
                    if key.startswith("Trace"):
                        plot.pop(key)
                # Update with renumbered traces
                plot.update(new_plot_traces)
                save_config()
                st.rerun()

            col_names = st.session_state.csv_df.columns.tolist()
            plot[trace]["x_col_in_csv"] = st.selectbox("X Axis", col_names, index=col_names.index(plot[trace]["x_col_in_csv"]))
            plot[trace]["y_col_in_csv"] = st.selectbox("Y Axis", col_names, index=col_names.index(plot[trace]["y_col_in_csv"]))
            plot[trace]["color"] = st.color_picker("Color", plot[trace]["color"])
            plot[trace]["legend"] = st.text_input("Legend", plot[trace]["legend"])
            plot[trace]["line"] = st.selectbox("Line", ["solid", "dot", "dash"], index=["solid", "dot", "dash"].index(plot[trace]["line"]))
            plot[trace]["width"] = st.slider("Width", 1, 10, plot[trace]["width"])
            
        save_config()
        return

    
    # Plot the 2D graph
    fig = go.Figure()
    for i in range(1, len(plot)-9+1):
        trace = plot[f"Trace {i}"]
        if {trace["x_col_in_csv"], trace["y_col_in_csv"]}.issubset(st.session_state.csv_df.columns):
            fig.add_trace(go.Scatter(
                x= st.session_state.csv_df[trace["x_col_in_csv"]],
                y= st.session_state.csv_df[trace["y_col_in_csv"]],
                mode="markers" if trace["line"] == "dot" else "lines",
                line=dict(  color=trace["color"],
                            dash="dash" if trace["line"] == "dash" else "solid",
                            width=trace["width"] ),
                name=trace["legend"], ))
    if not plot["autoscale"]:
        fig.update_xaxes(range=[plot["x_min"], plot["x_max"]])
        fig.update_yaxes(range=[plot["y_min"], plot["y_max"]])
    fig.update_layout(
        title=plot["title"],
        showlegend=plot["showlegend"],
        xaxis_title=plot["xaxis_title"],
        yaxis_title=plot["yaxis_title"],
        margin=dict(l=0, r=0, b=0, t=0),
    )
    container.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    container.plotly_chart(fig, use_container_width=True, key=f"plot2d_{n_plot}")

def all_plot_2d():
    """Create the 2D plot section."""

    global config

    # Check if the CSV file is loaded
    if st.session_state.csv_df.empty:
        return

    # check if config["plot2d"] is defined
    if "plot2d" not in config:
        config["plot2d"] = []

    # check if config["plot2d"] is a list
    if not isinstance(config["plot2d"], list):
        config["plot2d"] = []
    
    nb_plot2d = len(config["plot2d"])
    i= 0
    while i < nb_plot2d+1:
        col_plot2d_l, col_plot2d_m, col_plot2d_r = st.columns([1, 1, 1])
        
        # Left plot
        if i < nb_plot2d:
            plot_2d_window(col_plot2d_l, i)
        else:
            col_plot2d_l.markdown("<div style='height:250px'></div>", unsafe_allow_html=True)
            _, col_plot2d_l_plus, _ = col_plot2d_l.columns([10, 2, 10])
            if col_plot2d_l_plus.button("➕", key="left"):
                config["plot2d"].append(create_default_plot2d())
                save_config()
                st.rerun()
            break
        i += 1

        # Middle plot
        if i < nb_plot2d:
            plot_2d_window(col_plot2d_m, i)
        else:
            col_plot2d_m.markdown("<div style='height:250px'></div>", unsafe_allow_html=True)
            _, col_plot2d_m_plus, _ = col_plot2d_m.columns([10, 2, 10])
            if col_plot2d_m_plus.button(" ➕ ", key="middle"):
                config["plot2d"].append(create_default_plot2d())
                save_config()
                st.rerun()
            break
        i += 1

        # Right plot
        if i < nb_plot2d:
            plot_2d_window(col_plot2d_r, i)
        else:
            col_plot2d_r.markdown("<div style='height:250px'></div>", unsafe_allow_html=True)
            _, col_plot2d_r_plus, _ = col_plot2d_r.columns([10, 2, 10])
            if col_plot2d_r_plus.button("➕", key="right"):
                config["plot2d"].append(create_default_plot2d())
                save_config()
                st.rerun()
            break
        i += 1

def space(pixels):
    """Create a space of given pixels."""
    return st.markdown("<div style='height:{}px'></div>".format(pixels), unsafe_allow_html=True)

def debug_show_session_state():
    if st.button("Show Session State (debug)"):
        st.write(st.session_state)

def main():
    """Main function to run the Streamlit app."""
    
    global config

    load_config()
        

    # application configuration
    st.set_page_config(layout="wide")

    # application content
    top_button_bar()
    space(200)
    if not st.session_state.csv_df.empty:
        title("2D Plots")
        csv_variables_table()
        debug_show_session_state()
        space(200)
        if show_3d_hopper_path:
            title("3D Animation")
            space(100)
            anim_3d()
            # 1. Lire et sauvegarder la caméra
            if st.button("📸 Save current camera view"):
                streamlit_js_eval(
                    js_expressions="""
                    let plot = document.querySelector(".js-plotly-plot");
                    if (plot && plot._fullLayout && plot._fullLayout.scene && plot._fullLayout.scene.camera) {
                        return plot._fullLayout.scene.camera;
                    }
                    return null;
                    """,
                    key="camera_view"
                )
            # 2. Stocker dans session_state
            camera = st.session_state.get("js_eval", None)
            if camera:
                st.session_state.saved_camera = camera
            # 3. Afficher ou réutiliser la caméra sauvegardée
            if "saved_camera" in st.session_state:
                st.write("Saved Camera:")
                st.json(st.session_state.saved_camera)

            space(20)
        title("2D Plots")
        space(100)
        all_plot_2d()
    else:
        st.info("No CSV file loaded...")
    space(400)
    
    save_config()




if __name__ == "__main__":
    main()

