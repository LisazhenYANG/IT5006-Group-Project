from pydeck.bindings import view_state
import streamlit as st
import pandas as pd

#import plotly.graph_objects as go # type: ignore
#import pydeck as pdk

# Configure Streamlit page layout
st.set_page_config(page_title="Chicago Crime (2015–2025)", layout="wide")


@st.cache_data(ttl=3600)
def load_crime() -> pd.DataFrame:
    """
    Load and clean Chicago crime data from local processed CSV file.
    Cleaning steps follow the preprocessing method shown in the notebook screenshot.
    """

    # ---------------------------
    # 1. Define local data path
    # ---------------------------
    DATA_PATH = "processed/chicago_crimes_2015_2024_cleaned.csv"

    # ---------------------------
    # 2. Load CSV file
    # ---------------------------
    df = pd.read_csv(DATA_PATH)

    # Convert 'date' column to datetime format for time-based analysis
    df["date"] = pd.to_datetime(df["date"])

    # ---------------------------
    # 3. Remove records without valid coordinates
    # ---------------------------
    df = df.dropna(subset=["latitude", "longitude"])

    # ---------------------------
    # 4. Inspect coordinate statistics
    # ---------------------------
    #st.write("### Coordinate Statistics")
    #st.write(df[["latitude", "longitude"]].describe())

    # ---------------------------
    # 4. Filter valid Chicago coordinates
    #    (Remove outliers outside reasonable geographic bounds)
    # ---------------------------
    df = df[(df["latitude"] > 37) & (df["latitude"] < 42) &
            (df["longitude"] > -91) & (df["longitude"] < -87)]

    # Display number of remaining records after filtering
    st.write(f"**Records after filtering:** {len(df):,}")

    return df

""" 
made by Group17
"""

# ---------------------------
# Main Dashboard Section
# ---------------------------

import plotly.express as px # type: ignore
import plotly.graph_objects as go # type: ignore
import pydeck as pdk

st.title("Chicago Crimes Dashboard (2015–2025)")

# --- Sidebar filters (interactive controls) ---
with st.sidebar:
    st.header("Filters")

    # Time range filter
    y1, y2 = st.slider("Year range", 2015, 2025, (2015, 2024))

    # Arrest filter
    arrest_filter = st.selectbox("Arrest", ["All", "True", "False"])

# --- Load data (already cleaned in load_crime) ---
df = load_crime()

# --- Convert/derive temporal fields used by the temporal notebook charts ---
df["year2"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["hour"] = df["date"].dt.hour
df["weekday"] = df["date"].dt.day_name()

# --- Apply year range filtering ---
df = df[(df["date"] >= f"{y1}-01-01") & (df["date"] <= f"{y2}-12-31 23:59:59")]

# --- Crime type filtering (multi-select) ---
types = sorted(df["primary_type"].dropna().unique().tolist())
selected_types = st.sidebar.multiselect(
    "Primary Type",
    types,
    default=types[:5] if len(types) >= 5 else types
)
if selected_types:
    df = df[df["primary_type"].isin(selected_types)]

# --- Arrest filtering (boolean conversion + filtering) ---
if arrest_filter != "All":
    df["arrest"] = df["arrest"].astype(str).str.lower().map({"true": True, "false": False})
    df = df[df["arrest"] == (arrest_filter == "True")]

# --- Show record count for transparency ---
st.caption(f"Records in view: {len(df):,}")

# ---------------------------
# Tab Layout (Temporal / Spatial / Spatiotemporal)
# ---------------------------
tab0,tab1, tab2, tab3, tab4 = st.tabs(["Overview","Temporal Patterns", "Spatial Distribution", "Spatiotemporal Insights","More Infomation"])

# ===========================
# TAB 0: Temporal Patterns
# ===========================

with tab0:
    st.subheader("Dataset Overview")

    # --- 1. KPI Metrics ---
    # Total number of crime records (based on current filtered view)
    total_records = len(df)

    # Count occurrences of each primary crime type
    vc = df["primary_type"].value_counts(dropna=True)

    # Identify most common crime type
    top_type = vc.index[0] if len(vc) > 0 else "N/A"
    top_type_count = int(vc.iloc[0]) if len(vc) > 0 else 0

    col1, col2 = st.columns(2)

    col1.metric(
        "Total Crime Records (Current View)",
        f"{total_records:,}"
    )

    col2.metric(
        "Most Common Crime Type",
        f"{top_type}",
        f"{top_type_count:,} records"
    )

    st.divider()

    left_col, right_col = st.columns(2)

    # --- 2. Crime Type Distribution (Pie Chart) ---
    with left_col:
        st.markdown("**Crime Type Composition**")

        top_n = 8  # Limit categories for better readability
        top_counts = vc.head(top_n)
        others_count = vc.iloc[top_n:].sum()

        if others_count > 0:
            top_counts = pd.concat(
                [top_counts, pd.Series({"Others": others_count})]
            )

        pie_df = top_counts.reset_index()
        pie_df.columns = ["primary_type", "count"]

        fig_pie = px.pie(
            pie_df,
            names="primary_type",
            values="count",
            title=f"Top {top_n} Crime Types + Others"
        )

        st.plotly_chart(fig_pie, use_container_width=True)

    # --- 3. Crime Records by Year (Line Chart) ---
    with right_col:
        st.markdown("**Crime Records by Year**")

        yearly_counts = (
            df.groupby("year2")
              .size()
              .rename("num_crimes")
              .reset_index()
              .sort_values("year2")
        )

        fig_year = px.line(
            yearly_counts,
            x="year2",
            y="num_crimes",
            markers=True,
            title="Yearly Crime Trend"
        )

        st.plotly_chart(fig_year, use_container_width=True)
        

# ===========================
# TAB 1: Temporal Patterns
# ===========================
with tab1:
    st.subheader("Temporal Patterns (Trend + Cycles + Heatmaps)")

    c1, c2 = st.columns([1, 1])

    with c1:
        # Monthly trend (long-term trend + seasonality)
        st.markdown("**Monthly Trend** (crime counts over time)")
        ts_m = df.set_index("date").resample("M").size().rename("num_crimes").reset_index()
        fig_m = px.line(ts_m, x="date", y="num_crimes", title="Monthly Crime Trend")
        st.plotly_chart(fig_m, width="stretch")

    with c2:
        # Hourly pattern (daily cycle)
        st.markdown("**Hourly Pattern** (0–23)")
        ts_h = df.groupby("hour").size().rename("num_crimes").reset_index()
        fig_h = px.bar(ts_h, x="hour", y="num_crimes", title="Hourly Crime Distribution")
        st.plotly_chart(fig_h, width="stretch")

    c3, c4 = st.columns([1, 1])

    with c3:
        # Month × Hour heatmap (seasonality × daily cycle) — core insight chart in the temporal notebook
        st.markdown("**Month × Hour Heatmap** (seasonality × daily cycle)")
        mxh = pd.crosstab(df["month"], df["hour"]).reindex(index=range(1, 13), columns=range(0, 24), fill_value=0)
        fig_mxh = go.Figure(
             data=go.Heatmap(z=mxh.values,x=mxh.columns,y=mxh.index,colorscale="RdBu",reversescale=True))

        fig_mxh.update_layout(title="Month × Hour Heatmap", xaxis_title="Hour", yaxis_title="Month")
        st.plotly_chart(fig_mxh, width="stretch")

    with c4:
        # Weekday × Hour heatmap (weekday structure × daily cycle)
        st.markdown("**Weekday × Hour Heatmap** (weekday structure × daily cycle)")
        weekday_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        wxh = pd.crosstab(df["weekday"], df["hour"]).reindex(index=weekday_order, columns=range(0, 24), fill_value=0)
        fig_wxh = go.Figure(data=go.Heatmap(z=wxh.values, x=wxh.columns, y=wxh.index,colorscale="RdBu",reversescale=True))
        fig_wxh.update_layout(title="Weekday × Hour Heatmap", xaxis_title="Hour", yaxis_title="Weekday")
        st.plotly_chart(fig_wxh, width="stretch")

# ===========================
# TAB 2: Spatial Distribution
# ===========================
with tab2:
    st.subheader("Spatial Distribution (Density + Composition)")
    st.markdown("**Spatial Density (Map Heatmap with place names)**")

    df_map = df[["latitude", "longitude"]].dropna()
    if len(df_map) > 250000:
        df_map = df_map.sample(250000, random_state=42)

    cell = 0.004
    df_map["lat_bin"] = (df_map["latitude"] / cell).round() * cell
    df_map["lon_bin"] = (df_map["longitude"] / cell).round() * cell

    grid = df_map.groupby(["lat_bin", "lon_bin"]).size().reset_index(name="count")

    vmax = int(grid["count"].quantile(0.98)) 
    vmax = max(vmax, 1)

    fig_den = px.density_mapbox(
        grid,
        lat="lat_bin",
        lon="lon_bin",
        z="count",
        radius=18,  
        zoom=9.8,
        center={"lat": 41.8781, "lon": -87.6298},
        mapbox_style="open-street-map",
        color_continuous_scale="Turbo",
        range_color=[0, vmax],
        opacity=0.8,
        hover_data={"lat_bin": ":.4f", "lon_bin": ":.4f", "count": True},
        )
    
    fig_den.update_layout(
        height=750,
        margin=dict(l=0, r=0, t=30, b=0),
        coloraxis_colorbar=dict(title="Count per grid cell"),
        )
    st.plotly_chart(fig_den, use_container_width=True)

# ===========================
# TAB 3: Spatiotemporal Insights
# ===========================
with tab3:
    st.subheader("Spatiotemporal Insights (Time Slice → Map & Trend)")
    st.markdown("**Select a single year to view spatial distribution and monthly trend**")

    year_pick = st.slider("Single year (slice)", int(df["year2"].min()), int(df["year2"].max()), int(df["year2"].max()))
    dfi = df[df["year2"] == year_pick].copy()
    st.caption(f"Records in selected year {year_pick}: {len(dfi):,}")

    c5, c6 = st.columns([2, 1], gap="large")

    with c5:
        st.markdown("**Crime Density Map (grid count, selected year)**")

        dfi_map = dfi[["latitude", "longitude"]].dropna()
        if len(dfi_map) > 250000:
            dfi_map = dfi_map.sample(250000, random_state=42)

        cell = 0.01  # fixed grid size (degree), roughly ~1km
        dfi_map["lat_bin"] = (dfi_map["latitude"] / cell).round() * cell
        dfi_map["lon_bin"] = (dfi_map["longitude"] / cell).round() * cell

        grid = dfi_map.groupby(["lat_bin", "lon_bin"]).size().reset_index(name="count")

        vmax = int(grid["count"].quantile(0.98))
        vmax = max(vmax, 1)

        fig_map = px.scatter_mapbox(
            grid,
            lat="lat_bin",
            lon="lon_bin",
            color="count",
            size="count",
            size_max=18,
            zoom=9.8,
            center={"lat": 41.8781, "lon": -87.6298},
            mapbox_style="open-street-map",
            color_continuous_scale="Turbo",
            range_color=[1, vmax],
            opacity=0.8,
            hover_data={"lat_bin": ":.4f", "lon_bin": ":.4f", "count": True},
        )
        fig_map.update_layout(
            height=520,
            margin=dict(l=0, r=0, t=40, b=0),
            coloraxis_colorbar=dict(title="Count per grid cell"),
        )
        st.plotly_chart(fig_map, width="stretch")

    with c6:
        st.markdown("**Monthly Trend (Selected Year)**")

        ts_year = (
            dfi.set_index("date")
               .resample("ME")
               .size()
               .rename("num_crimes")
               .reset_index()
        )
        fig_year = px.line(ts_year, x="date", y="num_crimes", title=f"Monthly Trend in {year_pick}")
        fig_year.update_layout(height=520, margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig_year, width="stretch")

# ===========================
# TAB 4: More information
# =========================== 
with tab4:
    st.subheader("Crime Structure & Risk Profile")

    dff = df.copy()
    if "arrest" in dff.columns:
        dff["arrest"] = dff["arrest"].astype(bool)
    if "domestic" in dff.columns:
        dff["domestic"] = dff["domestic"].astype(bool)

    top_n = 15
    top_types = dff["primary_type"].value_counts().head(top_n).index

    # 1) Arrest rate by crime type
    st.markdown(f"**Arrest Rate by Crime Type (Top {top_n} by volume)**")
    ar = (
        dff[dff["primary_type"].isin(top_types)]
        .groupby("primary_type")["arrest"]
        .mean()
        .mul(100)
        .sort_values(ascending=True)
        .reset_index(name="arrest_rate")
    )
    fig_ar = px.bar(ar, x="arrest_rate", y="primary_type", orientation="h", labels={"arrest_rate": "Arrest Rate (%)", "primary_type": "Crime Type"})
    fig_ar.update_layout(height=520, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_ar, width="stretch")

    # 2) Crime type × hour pattern (heatmap)
    st.markdown(f"**Crime Type × Hour Pattern**")
    hx = (
        dff[dff["primary_type"].isin(top_types)]
        .groupby(["primary_type", "hour"])
        .size()
        .reset_index(name="count")
    )
    fig_hx = px.density_heatmap(
        hx, x="hour", y="primary_type", z="count",
        histfunc="sum",
        labels={"hour": "Hour", "primary_type": "Crime Type", "count": "Count"},
        color_continuous_scale="RdBu_r"
    )
    fig_hx.update_layout(height=560, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_hx, width="stretch")

    # 3) Domestic vs non-domestic composition
    st.markdown(f"**Domestic vs Non-domestic (Top {top_n}Primary Type)**")
    if "domestic" in dff.columns:
        dom = (
            dff[dff["primary_type"].isin(top_types)]
            .groupby(["primary_type", "domestic"])
            .size()
            .reset_index(name="count")
        )
        fig_dom = px.bar(dom, x="primary_type", y="count", color="domestic", barmode="stack", labels={"primary_type": "Crime Type", "count": "Count", "domestic": "Domestic"})
        fig_dom.update_layout(height=520, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_dom, width="stretch")
    else:
        st.info("Column `domestic` not found in your dataset, so this plot is skipped.")

    # 4) Top locations
    st.markdown("**Location Hotspots (Top 15 Locations)**")
    if "location_description" in dff.columns:
        loc = (
            dff["location_description"]
            .value_counts()
            .head(15)
            .sort_values(ascending=True)
            .reset_index()
        )
        loc.columns = ["location_description", "count"]
        fig_loc = px.bar(loc, x="count", y="location_description", orientation="h", labels={"location_description": "Location", "count": "Count"})
        fig_loc.update_layout(height=560, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_loc, width="stretch")
    else:
        st.info("Column `location_description` not found in your dataset, so this plot is skipped.")