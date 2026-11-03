"""
TFB3133 Data Visualization - Lab 6: Population Distribution Map (Streamlit + GeoPandas + Folium)
Run:  streamlit run map_population.py
Needs malaysia_states.geojson in the same folder.
"""
import numpy as np
import pandas as pd
import geopandas as gpd
import folium
import streamlit as st
from streamlit_folium import folium_static

STATES = [
    "Johor", "Kedah", "Kelantan", "Melaka", "Negeri Sembilan",
    "Pahang", "Penang", "Perak", "Perlis", "Sabah", "Sarawak",
    "Selangor", "Terengganu", "Kuala Lumpur", "Labuan", "Putrajaya",
]

# GeoJSON files spell some states differently - map them to the names used in STATES
ALIASES = {
    "Federal Territory of Kuala Lumpur": "Kuala Lumpur", "Wilayah Persekutuan Kuala Lumpur": "Kuala Lumpur",
    "Federal Territory of Labuan": "Labuan", "Wilayah Persekutuan Labuan": "Labuan",
    "Federal Territory of Putrajaya": "Putrajaya", "Wilayah Persekutuan Putrajaya": "Putrajaya",
    "Pulau Pinang": "Penang", "Malacca": "Melaka", "Trengganu": "Terengganu",
}

# Task 2.4 - imaginary states (no shape in the GeoJSON, so given a point location instead)
IMAGINARY = {"Pulau Harapan": (6.9, 103.6), "Negeri Awan": (1.2, 109.0), "Tanah Tinggi": (3.0, 113.0)}


# ---------------------------------------------------------------- data (Task 2.1 + 2.2)
def generate_data(states, metric, low, high, seed):
    rng = np.random.default_rng(seed)
    values = rng.integers(low, high, size=len(states))
    return pd.DataFrame({"State": states, metric: values})


@st.cache_data
def load_map():
    gdf = gpd.read_file("malaysia_states.geojson")
    # find whichever column holds the state name ("name", "Name", "NAME_1"...)
    name_col = next(c for c in gdf.columns if c.lower() in ("name", "name_1", "state", "shapename"))
    gdf["name"] = gdf[name_col].replace(ALIASES)
    return gdf[["name", "geometry"]]


# ---------------------------------------------------------------- map (Task 2.3)
def plot_map(gdf, data, metric, palette, imaginary):
    m = folium.Map(location=[4.2105, 108.0], zoom_start=6, tiles="cartodbpositron")
    folium.Choropleth(
        geo_data=gdf, name=f"{metric} Distribution", data=data,
        columns=["State", metric], key_on="feature.properties.name",
        fill_color=palette, fill_opacity=0.75, line_opacity=0.4,
        nan_fill_color="lightgrey", legend_name=f"{metric} by state",
    ).add_to(m)

    # hover tooltip: state name + value
    merged = gdf.merge(data, left_on="name", right_on="State", how="left")
    merged[metric] = merged[metric].map(lambda v: f"{v:,.0f}" if pd.notna(v) else "no data")
    folium.GeoJson(
        merged, style_function=lambda f: {"fillOpacity": 0, "weight": 0},
        tooltip=folium.GeoJsonTooltip(fields=["name", metric], aliases=["State:", f"{metric}:"]),
    ).add_to(m)

    # imaginary states have no polygon -> draw them as circles sized by value
    vmax = data[metric].max()
    for name in imaginary:
        val = int(data.loc[data["State"] == name, metric].iloc[0])
        folium.CircleMarker(
            location=IMAGINARY[name], radius=8 + 20 * val / vmax, color="purple",
            fill=True, fill_opacity=0.6, tooltip=f"{name} (imaginary): {val:,}",
        ).add_to(m)
    return m


# ---------------------------------------------------------------- app
def main():
    st.set_page_config(page_title="Lab 6 - Malaysia Map", layout="wide")
    st.sidebar.header("Lab 6 controls")

    metric = st.sidebar.radio("Dataset (Task 2.2)", ["Population", "GDP"])
    if metric == "Population":
        preset = st.sidebar.radio("Range (Task 2.1)", ["0.5M - 5M (original)", "1M - 10M (modified)"])
        low, high = (500_000, 5_000_000) if preset.startswith("0.5") else (1_000_000, 10_000_000)
    else:
        low, high = 10_000, 100_000
    palette = st.sidebar.selectbox("Colour scheme (Task 2.3)", ["YlOrRd", "BuGn", "Blues", "Greens", "PuRd", "YlGnBu"])
    imaginary = st.sidebar.multiselect("Add imaginary states (Task 2.4)", list(IMAGINARY))
    drop = st.sidebar.multiselect("Remove real states (Task 2.4)", STATES)
    seed = st.sidebar.number_input("Random seed (same seed = same numbers)", 0, 9999, 42)

    states = [s for s in STATES if s not in drop] + imaginary
    data = generate_data(states, metric, low, high, seed)

    st.title(f"Malaysia {metric} Distribution Map")
    st.caption(f"Random {metric} between {low:,} and {high:,}  |  palette: {palette}  |  states: {len(states)}")

    gdf = load_map()
    missing = sorted(set(states) - set(gdf["name"]))
    if missing:
        st.info(f"No shape in the GeoJSON for: {', '.join(missing)} - shown as purple circles / not coloured.")

    c1, c2 = st.columns([1, 2.4])
    with c1:
        st.write(f"Sample {metric} data:")
        st.dataframe(data.sort_values(metric, ascending=False).style.format({metric: "{:,}"}),
                     hide_index=True, height=600)
        st.metric("Total", f"{data[metric].sum():,}")
    with c2:
        folium_static(plot_map(gdf, data, metric, palette, imaginary), width=900, height=600)


if __name__ == "__main__":
    main()