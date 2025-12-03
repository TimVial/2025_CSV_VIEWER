# 2025_CSV_VIEWER
An intuitive interface that allows you to plot data directly from CSV files, supporting multiple plots and multiple lines per plot for any variable combination. You can also create and visualize custom functions of your CSV variables. Everything is designed to be simple, efficient, and user-friendly.

## Installation and running

- Create and activate a virtual environment (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

- Install dependencies (or use `pip install -r requirements.txt` if you have one):
```powershell
pip install streamlit pandas plotly numpy numpy-stl streamlit-js-eval streamlit-modal matplotlib
```

- Run the Streamlit application:
```powershell
python -m streamlit run .\csv_viewer.py
```

- Open the app in your browser at: http://localhost:8501


If the default port is in use, Streamlit will print an alternate URL in the console; use that URL to access the app.


## Usage

This short guide explains how the app works: importing a CSV, adding plots, and configuring plot options.

1. Import a CSV

   - Click the **Load CSV** button at the top-left and select your `.csv` file.

   - Screenshot placeholder:

   ![Import CSV screenshot](docs/screenshots/import_csv.png)

   (Add a screenshot showing the `Load CSV` button and the file selection dialog.)

2. Add a plot

   - Once the CSV is loaded, use the `➕` button in the 2D plots area to create a new plot.

   - Screenshot placeholder:

   ![Add plot screenshot](docs/screenshots/add_plot.png)

   (Add a screenshot showing the `➕` button and the new empty plot panel.)

3. Plot options

   - For each plot, click the **⚙️** icon to open the plot options.
   - In the options window you can:
     - Edit the title, scales and limits (disable `Autoscale` to set `X min`, `X max`, `Y min`, `Y max`).
     - Manage traces (add/remove), choose CSV columns for `X Axis`/`Y Axis`, change color, line style and width.

   - Screenshot placeholder (options):

   ![Plot options screenshot](docs/screenshots/plot_options.png)

   (Add a screenshot showing the `Plot Options` window and the trace controls.)

Notes:

- CSV column names appear automatically in the `X Axis` and `Y Axis` selectors.
- You can create derived variables using the variables editor (section `Create new variables from existing CSV columns`) by writing expressions based on existing columns.

(Replace the images in `docs/screenshots/` with your real screenshots, or adjust the paths if you prefer a different folder.)

4. Function of CSV variables

5. Export and import configuration

6. Future upgrade and feature
- filter on data (attention compatibilité si la config est importée (importance de l'ordre de création des variables))
- 3d plots
