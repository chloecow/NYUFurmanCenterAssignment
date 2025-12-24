import pandas as pd
import geopandas as gpd
import numpy as np
import plotly.express as px
import plotly.io as pio
import re
from sodapy import Socrata
from pathlib import Path
from datetime import datetime

"""
Data cleaning / quality notes:
- Restricted data to April - September 2025
- 88 violation records with missing/ invalid community district identifiers for borocd column removed
- Complaint records were deduplicated s.t. there is one observation per complaint_id
- Borough names were standardized via uppercase for consistency
- Invalid community board numbers were coerced to null before district construction
- Violation records were limited to classes A, B, C
"""

## file paths
root_dir = Path(__file__).resolve().parent
data_dir = root_dir/"data"
shape_dir = data_dir/"nycd_25d"

## var for mapping borough names to NYC borough codes (for borocd ids)
boro_map = {
    "MANHATTAN": 1,
    "BRONX": 2,
    "BROOKLYN": 3,
    "QUEENS": 4,
    "STATEN ISLAND": 5,
    }

## var to create API client once
violations_id = "wvxf-dwi5" 
client = Socrata("data.cityofnewyork.us", None, timeout=90) 

## NYC OpenData -- Housing Maintenance Code Complaints and Problems

def load_complaints(path):
    '''
    load_complaints(path):
        params: path --> filepath for complaints data
        return: df of the complaints and problems data
    '''

    cols = ["Received Date", "Complaint ID", "Borough", "Community Board"]

    df = pd.read_csv(path, usecols=cols, low_memory=False).rename(columns={
        "Received Date": "received_date",
        "Complaint ID": "complaint_id",
        "Borough": "borough",
        "Community Board": "community_board",
    })

    print(f"raw complaints data: {len(df)} rows")

    # date filtering
    df["received_date"] = pd.to_datetime(df["received_date"], errors="coerce")
    invalid_dates = df["received_date"].isna().sum()
    print(f"Invalid dates: {invalid_dates}")

    before_filter=len(df)
    df = df[(df["received_date"] >= "2025-04-01") & (df["received_date"] < "2025-10-01")].copy()
    outside_range = before_filter - len(df)
    print(f"outside date range (April - September 2025): {outside_range}")

    # borocd creation
    df["borough"] = df["borough"].astype(str).str.upper().str.strip()
    df["community_board"] = pd.to_numeric(df["community_board"], errors="coerce").astype("Int64")
    df["boro_id"] = df["borough"].map(boro_map)

    # multiply by 100 to construct borocd in standardized code
    df["borocd"] = (df["boro_id"] * 100 + df["community_board"]).astype("Int64")
    df["month"] = df["received_date"].dt.to_period("M").astype(str)

    # tracking missing values
    before_dropna = len(df)
    missing_complaint_id = df["complaint_id"].isna().sum()
    missing_borocd = df["borocd"].isna().sum()

    # complaints+problems repeats complaint_id, keep one row per complaint
    df = df.dropna(subset=["complaint_id", "borocd"])
    after_dropna = len(df)

    print(f"missing complaint_id: {missing_complaint_id:,}")
    print(f"missing / invalid borocd: {missing_borocd}")
    print(f"total reoved (missing complaint_id / borocd): {before_dropna - after_dropna}")
    
    # tracking deduplication
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["complaint_id"])
    duplicate_count = before_dedup - len(df)
    print(f"duplicate complaint_ids removed: {duplicate_count}")
    
    print(f"Final complaints df: {len(df)} rows")
    return df

## NYC OpenData -- Housing Maintenance Code Violations api call
## note: chose to do API call because downloading this data was crashing my computer. 
## source for using Socrata: https://dev.socrata.com/docs/queries/

def fetch_violations():
    '''
    fetch_violations(): 
        params: none
        return: violations_df --> dataframe with violations data from April 2025 - September 2025
    '''

    # April - September 2025
    date_ranges = [
        ("2025-04-01", "2025-05-01"),
        ("2025-05-01", "2025-06-01"),
        ("2025-06-01", "2025-07-01"),
        ("2025-07-01", "2025-08-01"),
        ("2025-08-01", "2025-09-01"),
        ("2025-09-01", "2025-10-01"),
    ]

    dfs = []
    for start, end in date_ranges:
        where = (
            f"novissueddate >= '{start}T00:00:00.000' "
            f"AND novissueddate < '{end}T00:00:00.000'"
        )

        query = f"""
        SELECT
            violationid,
            novissueddate,
            class,
            boro,
            communityboard
        WHERE {where}
        LIMIT 200000
        """

        rows = client.get(violations_id, query=query) 
        df = pd.DataFrame.from_records(rows)
        dfs.append(df)

        print(f"{start} - {end}: {len(df)} rows")

    violations_df = pd.concat(dfs, ignore_index=True)
    print(f"initial violations loaded: {len(violations_df)} rows")

    # clean types and fields
    violations_df["novissueddate"] = pd.to_datetime(violations_df["novissueddate"], errors="coerce")
    violations_df["boro"] = violations_df["boro"].astype(str).str.upper().str.strip()
    violations_df["communityboard"] = pd.to_numeric(violations_df["communityboard"], errors="coerce").astype("Int64")
    violations_df["boro_id"] = violations_df["boro"].map(boro_map)
    violations_df["borocd"] = (violations_df["boro_id"] * 100 + violations_df["communityboard"]).astype("Int64")
    violations_df["month"] = violations_df["novissueddate"].dt.to_period("M").astype(str)
    
    # class filtering to only A/B/C
    violations_df["class"] = violations_df["class"].astype(str).str.upper().str.strip()
    before_class_filter = len(violations_df)
    violations_df = violations_df[violations_df["class"].isin(["A","B","C"])]
    non_abc_class = before_class_filter - len(violations_df)
    print(f"Non A/B/C violations removed: {non_abc_class}")

    # tracking missing borocd vals
    missing_borocd = violations_df["borocd"].isna().sum()
    print(f"Missing/ invalid borocd: {missing_borocd}")

    print(f"Final violations: {len(violations_df)} rows")
    return violations_df


def load_shapefile():
    '''
    load_shapefile():
        params: none
        return: the loaded community distrct shapefile data
    '''
    shp_path = shape_dir/"nycd.shp"
    gdf = gpd.read_file(shp_path)

    # rename column + correct type
    gdf["borocd"] = pd.to_numeric(gdf["BoroCD"], errors="coerce").astype("Int64")

    return gdf[["borocd", "geometry"]].copy()

## global var of regex pattern to find district header rows (e.g. "Bronx Community District 1", "Manhattan Community District 12")
district_pat = re.compile(
    r"(Manhattan|Bronx|Brooklyn|Queens|Staten Island)\s+Community District\s+(\d+)",
    re.IGNORECASE
)

# parse borough + district number into borocd
def to_borocd(district_name):
    m = district_pat.search(str(district_name))
    if not m:
        return np.nan

    # group 1 -- the borough name, group 2 -- the community district number
    boro_name = m.group(1).upper()
    cd_num = int(m.group(2))

    # map to borough codes
    boro_id = boro_map.get(boro_name, np.nan)

    return int(boro_id * 100 + cd_num)

def extract_metric(df, label_regex, value_col):
    '''
    extract_metric(df, label_regex, value_col):
        params:
            df: dataframe with 'district_name' already forward-filled
            label_regex: regex matching the row label, for example 'Total Population'
            value_col: column index of numeric value to extract for matched metric
        return: dataframe with one row per district_name and the extracted metric value 
    '''
    pattern = re.compile(label_regex, re.IGNORECASE)
    rows = []

    # loop through district name and rows to locate the metric row
    for dname, g in df.groupby("district_name", sort=False):
        label_match = (
            # match group 0 -- metric label in first column or group 1 -- metric label in second column
            g[0].astype(str).str.contains(pattern, na=False) |
            g[1].astype(str).str.contains(pattern, na=False)
        )

        row = g.loc[label_match].head(1)

        # extract numeric metric value if present, else nan
        val = pd.to_numeric(row.iloc[0, value_col], errors="coerce") if not row.empty else np.nan
        rows.append((dname, val))

    return pd.DataFrame(rows, columns=["district_name", "value"])
    

def load_census_data(path):
    '''
    load_census_data(path):
        path: filepath for census data
        return: df of census dataset
    '''
    raw_data = pd.read_excel(path, sheet_name=1, header=None)

    # forward fill district sections
    raw_data["district_name"] = raw_data[0].where(
        raw_data[0].astype(str).str.contains("Community District", na=False)
    )
    raw_data["district_name"] = raw_data["district_name"].ffill()

    # keep only rows inside district sections
    df = raw_data.dropna(subset=["district_name"]).copy()

    # 2010 Number = column 7, 2010 Percent = column 8
    col_2010_num = 7
    col_2010_pct = 8

    # base districts + borocd
    base = df[["district_name"]].drop_duplicates().copy()
    base["borocd"] = base["district_name"].apply(to_borocd).astype("Int64")

    # getting metrics -- raw strings used as regex patterns to match Census row labels
    pop = extract_metric(df, r"^Total Population$", col_2010_num).rename(columns={"value": "population"})
    pct_white = extract_metric(df, r"White Nonhispanic", col_2010_pct).rename(columns={"value": "pct_white_nh"})
    pct_black = extract_metric(df, r"Black Nonhispanic", col_2010_pct).rename(columns={"value": "pct_black_nh"})
    pct_hisp  = extract_metric(df, r"Hispanic Origin", col_2010_pct).rename(columns={"value": "pct_hispanic"})
    pct_asian = extract_metric(df, r"Asian.*Nonhispanic|Asian and Pacific Islander", col_2010_pct).rename(columns={"value": "pct_asian_pi"})

    out = (
        base.merge(pop, on="district_name", how="left")
            .merge(pct_white, on="district_name", how="left")
            .merge(pct_black, on="district_name", how="left")
            .merge(pct_hisp,  on="district_name", how="left")
            .merge(pct_asian, on="district_name", how="left")
    )

    # clean + return (typecast int, remove rows where borocd / population is missing)
    out["population"] = pd.to_numeric(out["population"], errors="coerce")
    out = out.dropna(subset=["borocd", "population"]).drop_duplicates(subset=["borocd"])

    return out[["borocd", "population", "pct_white_nh", "pct_black_nh", "pct_hispanic", "pct_asian_pi"]]

def clean_violations(violations):
    '''
        clean_violations(violations):
            params: violations -> df of violation data
            return: df of violations with class complaints only A, B, or C
            (according to source: https://www.nyc.gov/site/hpd/services-and-information/penalties-and-fees.page)
    '''
    v = violations.copy()
    v["class"] = v["class"].astype(str).str.upper().str.strip()
    return v[v["class"].isin(["A", "B", "C"])]

def complaints_by_district(complaints):
    '''
        complaints_by_district(complaints):
            params: complaints -> df of complaints data
            return: df of unique complaint counts agg by community district (borocd)
    '''
    complaints_district = (complaints.dropna(subset=["borocd", "complaint_id"])
        .groupby("borocd")["complaint_id"].nunique()
        .reset_index(name="complaints")
        .sort_values("complaints", ascending=False))
    
    return complaints_district

def complaints_by_month(complaints):
    '''
        complaints_by_month(complaints):
            params: complaints -> df of complaints data
            return: df of unique complaint counts agg by month and community district (borocd)
    '''
    complaints_month = (complaints.dropna(subset=["borocd", "complaint_id"])
        .groupby(["month", "borocd"])["complaint_id"].nunique()
        .reset_index(name="complaints")
        .sort_values(["borocd", "month"])
    )

    return complaints_month

def violations_by_district(violations):
    '''
        violations_by_district(violations):
            params: violations -> df of violations data
            return: df of unique violation counts agg by community district (borocd)
    '''
    violations_district = (violations.dropna(subset=["borocd"])
        .groupby("borocd").size()
        .reset_index(name="violations")
        .sort_values("violations", ascending=False)
    )

    return violations_district

def violations_by_month(violations):
    '''
        violations_by_month(violations):
            params: violations -> df of violations data
            return: df of unique complaint counts agg by month and community district (borocd)
    '''
    
    violations_month = (violations.dropna(subset=["month", "borocd"])
        .groupby(["month", "borocd"])
        .size()
        .reset_index(name="violations")
        .sort_values(["borocd", "month"])
    )

    return violations_month

def census_summary(census, complaints_by_district, violations_by_district):
    '''
        census_summary(census, complaints_by_district, violations_by_district)
            params: 
                census: df of Census demographic data indexed by community district (borocd),
                        includes population and demographic indicators
                complaints_by_district: df of aggregated complaint counts by community district
                violations_by_district: df of aggregated violation counts by community district
            return: df combining Census data with complaint and violation counts and
                    calculated per-capita / ratio metrics
    '''

    # merge Census demographics with aggregated complaint and violation counts by community district
    summary = (
        census.merge(complaints_by_district, on="borocd", how="left")
              .merge(violations_by_district, on="borocd", how="left")
    )

    # replace missing vals with 0
    summary["complaints"] = summary["complaints"].fillna(0).astype(int)
    summary["violations"] = summary["violations"].fillna(0).astype(int)

    # calculating per-capita complaint and violation rates per 1000 residents
    summary["complaints_per_1k"] = (summary["complaints"] / summary["population"]) * 1000
    summary["violations_per_1k"] = (summary["violations"] / summary["population"]) * 1000
    summary["violations_per_complaint"] = np.where(
        summary["complaints"] > 0,
        summary["violations"] / summary["complaints"],
        np.nan
    )

    return summary

def demog_correlations(summary, rate_col):
    '''
        demog_correlations(summary):
            params: 
                summary -> df with complaints per capita and demographic percentages
                rate_col -> column name of the desired per-capita rate
                          ("complaints_per_1k" / "violations_per_1k")
            return: series showing correlations between complaints per capita and demographic indicators
    '''

    demog_cols = ["pct_white_nh", "pct_black_nh", "pct_hispanic", "pct_asian_pi"]
    
    correlations = {}
    for col in demog_cols:
        correlations[col] = summary[rate_col].corr(summary[col])

    correlations_df = pd.Series(correlations).sort_values(ascending=False)
    return correlations_df

def plot_temporal_trends(complaints_by_month, top_districts, outpath):
    '''
        plot_temporal_trends(complaints_by_month, top_districts, outpath):
            params:
                complaints_by_month: df of aggregated complaint counts by community district and month
                top_districts: list of community district codes (borocd) to plot
                outpath: file path to save the output plot
            return: nothing, saves and displays graph
    '''
    
    # choosing top districts (the community districts with the highest total number of complaints) in complaints by month df
    df = complaints_by_month[complaints_by_month["borocd"].isin(top_districts)].copy()

    # convert month to date time so months are sorted
    df["month"] = pd.to_datetime(df["month"] + "-01")
    df = df.sort_values(["borocd", "month"])

    # source: https://plotly.com/python/line-charts/
    fig = px.line(
        df,
        x="month",
        y="complaints",
        color="borocd",
        markers=True,
        title="Housing Complaints in NYC Districts from April - September 2025 (Top 5 Districts with Most Complaints)",
        labels={"month": "Month", "complaints": "Number of Complaints", "borocd": "Community District"},
    )

    fig.update_layout(
        xaxis_tickformat="%Y-%m",
        xaxis_tickangle=-45,
        legend_title_text="District (BoroCD)",
        template="plotly_white",
        margin=dict(l=40, r=40, t=70, b=120),
    )
    fig.add_annotation(
        text=("Source: NYC Open Data - Housing Maintenance Code Complaints and Problems"),
        xref="paper", yref="paper",
        x=0, y=-0.22,
        showarrow=False,
        font=dict(size=11, color="gray"),
        align="left"
    )

    # saving plot
    outpath = str(outpath)
    fig.write_html(outpath + ".html", include_plotlyjs="cdn")

    fig.show()

def plot_violations_temporal_trends(violations_by_month, top_districts, outpath):
    '''
        plot_violations_temporal_trends(violations_by_month, top_districts, outpath):
            params:
                violations_by_month: df of aggregated violation counts by community district and month
                top_districts: list of community district codes (borocd) to plot
                outpath: file path to save the output plot
            return: nothing, saves and displays graph
    '''

    # choose top districts
    df = violations_by_month[violations_by_month["borocd"].isin(top_districts)].copy()

    # convert month to datetime so it sorts correctly
    df["month"] = pd.to_datetime(df["month"] + "-01")
    df = df.sort_values(["borocd", "month"])

    fig = px.line(
        df,
        x="month",
        y="violations",
        color="borocd",
        markers=True,
        title="Housing Code Violations in NYC Districts from April - September 2025 (Top 5 Districts with Most Violations)",
        labels={
            "month": "Month",
            "violations": "Number of Violations",
            "borocd": "Community District (BoroCD)"
        },
    )

    fig.update_layout(
        xaxis_tickformat="%Y-%m",
        xaxis_tickangle=-45,
        legend_title_text="District (BoroCD)",
        template="plotly_white",
        margin=dict(l=40, r=40, t=70, b=120),
    )

    fig.add_annotation(
        text="Source: NYC Open Data - Housing Maintenance Code Violations",
        xref="paper",
        yref="paper",
        x=0,
        y=-0.22,
        showarrow=False,
        font=dict(size=11, color="gray"),
        align="left"
    )

    outpath = str(outpath)
    fig.write_html(outpath + ".html", include_plotlyjs="cdn")
    fig.show()

def plot_complaints_map(districts, summary, outpath):
    '''
        plot_complaint_map(districts, summary, outpath):
            params:
                districts: geodf of NYC community district geometries
                summary: df with complaints per capita and demographic percentages
                outpath: file path to save the output plot
            return: nothing, saves and displays graph
    '''
    
    gdf = districts.merge(summary, on="borocd", how="left").copy()

    # convert to WGS84 (lat/lon)
    if gdf.crs is not None:
       gdf = gdf.to_crs(epsg=4326)

    # build geojson from geodataframe
    geojson = gdf.__geo_interface__

    # source: https://plotly.com/python/choropleth-maps/
    fig = px.choropleth(
        gdf,
        geojson=geojson,
        locations=gdf.index,
        color="complaints_per_1k",
        hover_name="borocd",
        hover_data={"complaints_per_1k": True, "population": True, "complaints": True, "borocd": False},
        color_continuous_scale="YlOrRd",
        title="Housing Complaints per Capita by Community District<br><sup>April–September 2025</sup>",
        labels={"complaints_per_1k": "Complaints per 1,000 Residents"},
    )
    
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=10, r=10, t=90, b=120),
        coloraxis_colorbar=dict(title="Per 1,000 Residents"),
    )

    fig.add_annotation(
        text=(
            "Source: NYC Open Data — Housing Maintenance Code Complaints and Problems; "
            "NYC Community District Boundaries; "
            "Census Demographics at the NYC Community District level"
        ),
        xref="paper", yref="paper",
        x=0, y=-0.20,
        showarrow=False,
        font=dict(size=11, color="gray"),
        align="left"
    )

    outpath = str(outpath)
    fig.write_html(outpath + ".html", include_plotlyjs="cdn")

    fig.show()

if __name__ == "__main__":
    '''
        main():
            params: none
            return: none --> data loading, analysis, and visualization output
    '''

    # filepaths
    complaints_path = data_dir/"Housing_Maintenance_Code_Complaints_and_Problems_20251222.csv"
    census_path = data_dir/"sf1_dp_cd_demoprofile.xlsx"

    print("Load complaints and problems data")
    complaints = load_complaints(complaints_path)

    print("\nLoad violations data")
    violations = clean_violations(fetch_violations())

    print("\nLoad community district shapefile")
    districts = load_shapefile()

    print("\nLoad Census demographics data")
    census = load_census_data(census_path)

    print("\nBasic checks:")
    print("Complaints shape:", complaints.shape)
    print("Violations shape:", violations.shape)
    print("Districts shape:", districts.shape)
    print("\nSample Census data:")
    print(census.head())

    # ---- aggregates ----
    complaints_by_district = complaints_by_district(complaints)
    complaints_by_month = complaints_by_month(complaints)
    violations_by_district = violations_by_district(violations)
    violations_by_month = violations_by_month(violations)

    # ---- summary table (all per-capita metrics) ----
    summary = census_summary(census, complaints_by_district, violations_by_district)

    # ---- Q1 ----
    print("\nQ1: Top districts by complaints:")
    print(complaints_by_district.head(10))

    print("\nQ1(a): Top complaints per capita:")
    print(summary.sort_values("complaints_per_1k", ascending=False).head(10)[
        ["borocd", "population", "complaints", "complaints_per_1k"]
    ])

    print("\nQ1(b): Complaints by month -- first 10 rows:")
    print(complaints_by_month.head(10))

    print("\nQ1(c): Correlation with demographic indicators:")
    print(demog_correlations(summary, "complaints_per_1k").to_string())

    # ---- Q2 ----
    print("\nQ2: Top districts by violations:")
    print(violations_by_district.head(10))

    print("\nQ2(a): Top violations per capita:")
    print(summary.sort_values("violations_per_1k", ascending=False).head(10)[
        ["borocd", "population", "violations", "violations_per_1k"]
    ])

    print("\nQ2(b): Class violations counts (A/B/C):")
    violations_by_class = (
        violations["class"]
        .value_counts()
        .rename_axis("class")
        .reset_index(name="count")
    )

    violations_by_class["percent"] = (
        violations_by_class["count"] / violations_by_class["count"].sum() * 100
    )

    print(violations_by_class.to_string(index=False))

    print("\nQ2(c): Temporal trends of violations -- first 10 rows:")
    print(violations_by_month.head(10))

    print("\nQ2(d): Highest violations per complaint -- first 10 rows:")
    print(summary.sort_values("violations_per_complaint", ascending=False).head(10)[
        ["borocd", "complaints", "violations", "violations_per_complaint"]
    ])

    print("\nQ2(e): Correlation with demographic indicators:")
    print(demog_correlations(summary, "violations_per_1k").to_string())

    # ---- plots ----
    top5 = complaints_by_district.head(5)["borocd"].tolist()
    top5_violations = violations_by_district.head(5)["borocd"].tolist()
    plot_temporal_trends(complaints_by_month, top5, data_dir/"complaints_temporal_trends.png")
    plot_violations_temporal_trends(violations_by_month, top5_violations, data_dir/"violations_temporal_trends")
    plot_complaints_map(districts, summary, data_dir/"complaints_per_capita_map.png")