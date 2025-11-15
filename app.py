import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
# from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Mutual Fund Dashboard", layout="wide")


# ================================================================
# AUTO-DETECT HEADER ROW
# ================================================================
def detect_header_row(df):
    """
    Detect the best header row by checking:
    - at least 3 non-null values
    - at least 3 cells containing strings
    """
    for i in range(len(df)):
        row = df.iloc[i]
        if row.notnull().sum() >= 3 and row.apply(lambda x: isinstance(x, str)).sum() >= 3:
            return i
    return None


# ================================================================
# LOAD FILE WITH AUTO + MANUAL HEADER HANDLING
# ================================================================
def load_file(uploaded):
    if uploaded.name.lower().endswith(".csv"):
        df_temp = pd.read_csv(uploaded, header=None)
    else:
        df_temp = pd.read_excel(uploaded, header=None)

    header_row = detect_header_row(df_temp)

    if header_row is None:
        st.warning("⚠ Unable to auto-detect column headers. Please select manually.")
        st.write("Preview of rows:")
        st.dataframe(df_temp.head(10))

        header_row = st.number_input(
            "Select header row (0–10):", min_value=0, max_value=10, value=0
        )

    if uploaded.name.lower().endswith(".csv"):
        df = pd.read_csv(uploaded, header=header_row)
    else:
        df = pd.read_excel(uploaded, header=header_row)

    return df


# ================================================================
# NORMALIZE COLUMN NAMES
# ================================================================
def normalize_columns(df):

    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
        .str.replace(r"[()%*]", "", regex=True)
        .str.replace("__", "_")
        .str.replace(r"_$", "", regex=True)
        .str.replace(r"\.$", "", regex=True)
    )

    # Auto detect AUM column
    for col in df.columns:
        if "aum" in col and ("cr" in col or "crore" in col):
            df = df.rename(columns={col: "daily_aum_cr"})

    # Fix returns 1Y/3Y/5Y/10Y
    for yr in ["1", "3", "5", "10"]:
        df = df.rename(columns={
            f"return_{yr}_year_regular": f"return_{yr}_year_regular",
            f"return_{yr}_year__regular": f"return_{yr}_year_regular",
            f"return_{yr}_year_direct": f"return_{yr}_year_direct",
            f"return_{yr}_year__direct": f"return_{yr}_year_direct",
            f"return_{yr}_year_benchmark": f"return_{yr}_year_benchmark",
            f"return_{yr}_year__benchmark": f"return_{yr}_year_benchmark",
        })

    # Fix IR columns
    for yr in ["1", "3", "5"]:
        df = df.rename(columns={
            f"information_ratio_{yr}_year_regular": f"information_ratio_{yr}_year_regular",
            f"information_ratio_{yr}_year_direct": f"information_ratio_{yr}_year_direct",
        })

    df = df.rename(columns={
        "return_since_launch_regular": "return_since_launch_regular",
        "return_since_launch_direct": "return_since_launch_direct",
        "return_since_launch__benchmark": "return_since_launch_benchmark",
        "return_since_launch_benchmark": "return_since_launch_benchmark",
        "return_since_launch_direct_benchmark": "return_since_launch_direct_benchmark",
    })

    return df


# ================================================================
# FULL ANALYSIS
# ================================================================
def min_max_scale(series):
    series = series.astype(float)
    return (series - series.min()) / (series.max() - series.min() + 1e-9)

def run_analysis(df):

    # Convert numeric safely
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")

    # ----------------------------------------------------
    # AUM vs Returns Scatter (with Fund Name on hover)
    # ----------------------------------------------------
    st.subheader("📈 AUM vs Returns")

    needed = ["scheme_name", "daily_aum_cr", "return_1_year_regular", "return_3_year_regular", "return_5_year_regular"]

    if all(c in df.columns for c in needed):

        fig_scatter = px.scatter(
            df,
            x="daily_aum_cr",
            y="return_3_year_regular",
            hover_name="scheme_name",     # <<< FUND NAME IN TOOLTIP
            hover_data={
                "daily_aum_cr": True,
                "return_1_year_regular": True,
                "return_3_year_regular": True,
                "return_5_year_regular": True
            },
            title="AUM vs 3-Year Returns"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.warning("Missing columns for scatter plot.")

    # ----------------------------------------------------
    # Top 10 funds by 3-year returns
    # ----------------------------------------------------
    if "return_3_year_regular" in df.columns:
        st.subheader("🏆 Top 10 Funds by 3-Year Returns")
        top3 = df.sort_values("return_3_year_regular", ascending=False).head(10)
        fig_top3 = px.bar(top3, x="scheme_name", y="return_3_year_regular", text_auto=True)
        fig_top3.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_top3, use_container_width=True)

    # ----------------------------------------------------
    # Top 10 funds by 5-year Information Ratio
    # ----------------------------------------------------
    if "information_ratio_5_year_regular" in df.columns:
        st.subheader("📊 Top 10 Funds by 5-Year Information Ratio")
        top_ir = df.sort_values("information_ratio_5_year_regular", ascending=False).head(10)
        fig_ir = px.bar(top_ir, x="scheme_name", y="information_ratio_5_year_regular", text_auto=True)
        fig_ir.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_ir, use_container_width=True)

    # ----------------------------------------------------
    # Consistent Outperformers
    # ----------------------------------------------------
    st.subheader("🔥 Consistent Outperformers (1Y, 3Y, 5Y)")

    if (
        "return_1_year_regular" in df.columns and
        "return_1_year_benchmark" in df.columns and
        "return_3_year_regular" in df.columns and
        "return_3_year_benchmark" in df.columns and
        "return_5_year_regular" in df.columns and
        "return_5_year_benchmark" in df.columns
    ):
        consistent = df[
            (df["return_1_year_regular"] > df["return_1_year_benchmark"]) &
            (df["return_3_year_regular"] > df["return_3_year_benchmark"]) &
            (df["return_5_year_regular"] > df["return_5_year_benchmark"])
        ]

        if not consistent.empty:
            fig_con = px.bar(
                consistent,
                x="scheme_name",
                y=["return_1_year_regular", "return_3_year_regular", "return_5_year_regular"],
                barmode="group",
                title="Consistent Outperformers Returns",
                text_auto=True
            )
            fig_con.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_con, use_container_width=True)
        else:
            st.info("No consistent outperformers found.")

    # ----------------------------------------------------
    # SCORING MODEL
    # ----------------------------------------------------
    st.subheader("🏅 Composite Scoring Model")

    scaler = MinMaxScaler()

    perf_cols = ["return_1_year_regular", "return_3_year_regular", "return_5_year_regular"]
    ir_cols = ["information_ratio_1_year_regular", "information_ratio_3_year_regular", "information_ratio_5_year_regular"]

    if all(c in df.columns for c in perf_cols):
        #df["performance_score"] = scaler.fit_transform(df[perf_cols].mean(axis=1).values.reshape(-1, 1))
        df["performance_score"] = min_max_scale(df[perf_cols].mean(axis=1))
    else:
        df["performance_score"] = 0

    if all(c in df.columns for c in ir_cols):
        #df["ir_score"] = scaler.fit_transform(df[ir_cols].mean(axis=1).values.reshape(-1, 1))
        df["ir_score"] = min_max_scale(df[ir_cols].mean(axis=1))
    else:
        df["ir_score"] = 0

    df["consistency_score"] = (
        (df.get("return_1_year_regular", 0) > df.get("return_1_year_benchmark", 0)) &
        (df.get("return_3_year_regular", 0) > df.get("return_3_year_benchmark", 0)) &
        (df.get("return_5_year_regular", 0) > df.get("return_5_year_benchmark", 0))
    ).astype(int)

    if "daily_aum_cr" in df.columns:
        df["aum_score"] = 1 - np.abs(
            (df["daily_aum_cr"] - df["daily_aum_cr"].median()) / df["daily_aum_cr"].max()
        )
    else:
        df["aum_score"] = 0

    df["final_score"] = (
        df["performance_score"] * 0.4 +
        df["ir_score"] * 0.4 +
        df["consistency_score"] * 0.1 +
        df["aum_score"] * 0.1
    ) * 100

    ranked = df.sort_values("final_score", ascending=False)

    st.dataframe(
        ranked[["scheme_name", "final_score","return_1_year_regular","return_3_year_regular","return_5_year_regular"]].reset_index(drop=True),
        use_container_width=True
    )


# ================================================================
# STREAMLIT UI FLOW
# ================================================================
st.title("📊 Mutual Fund Performance Dashboard")

uploaded = st.file_uploader("Upload Excel/CSV Mutual Fund File", type=["xlsx", "xls", "csv"])

# if not uploaded:
#     st.info("Upload a file to begin. - Download link --> https://www.amfiindia.com/otherdata/fund-performance \n "
#             "Sample File: https://github.com/nilesh604/mf-data-automation/blob/main/Fund-Performance-15-Nov-2025--1049.xlsx")
#     st.stop()
if not uploaded:
    st.info("""
### 📤 Upload a Mutual Fund File to Begin

You can download the official AMFI fund performance file here:  
🔗 **[AMFI Fund Performance Download](https://www.amfiindia.com/otherdata/fund-performance)**  

Sample file for testing:  
🔗 **[Sample Excel File](https://github.com/nilesh604/mf-data-automation/blob/main/Fund-Performance-15-Nov-2025--1049.xlsx)**  
""")
    st.stop()

df_raw = load_file(uploaded)

if df_raw is None or df_raw.empty:
    st.error("❌ Could not load file. Try a different Excel/CSV.")
    st.stop()

df = normalize_columns(df_raw)

st.subheader("🔍 Data Preview")
st.dataframe(df.head(), use_container_width=True)

if st.button("🚀 Run Analysis"):
    run_analysis(df)
