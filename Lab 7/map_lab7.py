"""
TFB3133 Data Visualization - Lab 7: Interactive Map Visualization Examples
Run:  streamlit run map_lab7.py
Needs folder: ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp (+ .shx .dbf .prj)
"""
import streamlit as st
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
HERE = Path(__file__).parent

st.set_page_config(page_title="Lab 7 - Map Visualization", layout="wide")


# ---------------------------------------------------------------- data
@st.cache_data
def load_data():
    gdf = gpd.read_file(HERE / "ne_110m_admin_0_countries" / "ne_110m_admin_0_countries.shp")
    # Natural Earth uses UPPERCASE columns (NAME, POP_EST) - lower-case them so the
    # lab's 'pop_est' check finds the REAL population instead of making random numbers
    gdf.columns = [c.lower() if c != "geometry" else c for c in gdf.columns]
    gdf = gdf.set_geometry("geometry")
    if "pop_est" not in gdf.columns:
        np.random.seed(0)
        gdf["pop_est"] = np.random.randint(1e6, 1e9, size=len(gdf))
    # density in people per km2: area must be measured in an equal-area CRS (metres),
    # not in degrees (EPSG:4326), or countries near the poles get the wrong area
    area_km2 = gdf.to_crs("ESRI:54009").geometry.area / 1e6
    gdf["pop_density"] = gdf["pop_est"] / area_km2
    return gdf


gdf = load_data()
country_col = "name"
CMAPS = ["OrRd", "YlGn", "Blues", "Purples", "Greens"]

# the lab code uses projection_options but never defines it - defined here
projection_options = {
    "Plate Carree (WGS84, EPSG:4326)": "EPSG:4326",
    "Mercator (EPSG:3857)": "EPSG:3857",
    "Robinson (ESRI:54030)": "ESRI:54030",
    "Mollweide - equal area (ESRI:54009)": "ESRI:54009",
    "Equal Earth (EPSG:8857)": "EPSG:8857",
}


def density_column(use_log):
    """Return the column to plot + legend label."""
    if use_log:
        gdf["plot_val"] = np.log1p(gdf["pop_density"])
        return "plot_val", "log(1 + people per km²)"
    gdf["plot_val"] = gdf["pop_density"]
    return "plot_val", "people per km²"


# ---------------------------------------------------------------- sidebar
st.title("Interactive Map Visualization Examples")
st.sidebar.header("Choose a Map Type")
map_type = st.sidebar.radio("Select Map Type", [
    "Map Cosmetics", "Map Visual Hierarchy", "Choropleth Map", "Geo Projection"])

# ---------------------------------------------------------------- 1 Map cosmetics
if map_type == "Map Cosmetics":
    c1, c2, c3 = st.columns(3)
    fill = c1.color_picker("Land colour", "#87CEEB")          # skyblue
    bg = c2.color_picker("Ocean / background colour", "#D3D3D3")  # lightgrey
    lw = c3.slider("Border width", 0.0, 2.0, 0.5, 0.1)
    fig, ax = plt.subplots(figsize=(12, 6))
    gdf.plot(ax=ax, color=fill, edgecolor="black", linewidth=lw)
    ax.set_title("Styled Map Example", fontsize=15, fontweight="bold")
    ax.set_facecolor(bg)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    st.pyplot(fig)

# ---------------------------------------------------------------- 2 Visual hierarchy
elif map_type == "Map Visual Hierarchy":
    highlighted = st.multiselect(
        "Select countries to highlight in red", sorted(gdf[country_col].unique()),
        default=["Brazil", "India", "China", "Australia", "South Africa", "Russia", "Malaysia"])
    gdf["color"] = gdf[country_col].apply(lambda x: "red" if x in highlighted else "lightgrey")
    fig, ax = plt.subplots(figsize=(12, 6))
    gdf.plot(color=gdf["color"], edgecolor="white", linewidth=0.5, ax=ax)
    # label highlighted countries at a point guaranteed inside the shape
    for _, r in gdf[gdf[country_col].isin(highlighted)].iterrows():
        p = r.geometry.representative_point()
        ax.annotate(r[country_col], (p.x, p.y), ha="center", fontsize=8, fontweight="bold")
    ax.set_title("Map Visual Hierarchy with Selected Countries Highlighted in Red")
    ax.set_axis_off()
    st.pyplot(fig)

# ---------------------------------------------------------------- 3 Choropleth
elif map_type == "Choropleth Map":
    color_choice = st.selectbox("Choose Color Scheme", CMAPS, index=0)
    use_log = st.checkbox("Use Log Transformation", value=True)
    col, label = density_column(use_log)
    fig, ax = plt.subplots(figsize=(12, 6))
    gdf.plot(column=col, cmap=color_choice, edgecolor="black", linewidth=0.3, legend=True,
             legend_kwds={"label": label, "shrink": 0.6}, ax=ax)
    ax.set_title(f"Population Density by Country (Color Scheme: {color_choice})")
    ax.set_axis_off()
    st.pyplot(fig)
    st.write("Top 10 most dense countries:")
    st.dataframe(gdf.nlargest(10, "pop_density")[[country_col, "pop_est", "pop_density"]]
                 .round(1), hide_index=True)

# ---------------------------------------------------------------- 4 Geo projection
elif map_type == "Geo Projection":
    proj_choice = st.selectbox("Select a Projection", list(projection_options.keys()))
    color_choice = st.selectbox("Choose Color Scheme", CMAPS, index=0)
    use_log = st.checkbox("Use Log Transformation for Population Density", value=True)
    col, label = density_column(use_log)
    data = gdf
    if "Mercator" in proj_choice:
        # Mercator stretches to infinity at the poles - drop Antarctica
        data = gdf[gdf[country_col] != "Antarctica"]
    gdf_proj = data.to_crs(projection_options[proj_choice])
    fig, ax = plt.subplots(figsize=(12, 6))
    gdf_proj.plot(column=col, cmap=color_choice, edgecolor="black", linewidth=0.3, legend=True,
                  legend_kwds={"label": label, "shrink": 0.6}, ax=ax)
    ax.set_title(f"Map in {proj_choice} Projection")
    ax.set_axis_off()
    st.pyplot(fig)
