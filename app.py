import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from io import BytesIO
from math import ceil

st.set_page_config(page_title="PTR Explorer — Province Compare & Prediction", layout="wide")

st.title("📊 Teacher–Pupil Ratio (PTR) — Compare & Predict")
st.write("Upload your dataset, map columns, explore province-level ratios, compare two provinces, and **predict teachers needed** for a given student intake.")

# ---------------- Sidebar: Upload ----------------
st.sidebar.header("1) Load Data")
file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])

@st.cache_data(show_spinner=False)
def load_data(f):
    name = f.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(f)
    return pd.read_excel(f)

if not file:
    st.info("Upload a CSV/XLSX file in the sidebar to begin.")
    st.stop()

try:
    df = load_data(file)
except Exception as e:
    st.error(f"Could not read file: {e}")
    st.stop()

df.columns = df.columns.astype(str)

# ---------------- Dataset Details ----------------
with st.expander("🧾 Dataset details", expanded=False):
    c1, c2 = st.columns([1,1])
    with c1:
        st.write("**Shape:**", df.shape)
        st.write("**Columns:**", list(df.columns))
        dtypes = pd.DataFrame({"column": df.columns, "dtype": df.dtypes.astype(str)})
        st.dataframe(dtypes, use_container_width=True, hide_index=True)
    with c2:
        miss = df.isna().sum().reset_index().rename(columns={"index":"column", 0:"missing"})
        st.write("**Missing values per column:**")
        st.dataframe(miss, use_container_width=True, hide_index=True)
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if num_cols:
        st.write("**Numeric summary (first 8 numeric cols):**")
        st.dataframe(df[num_cols[:8]].describe().T, use_container_width=True)

# ---------------- Column Mapping ----------------
st.sidebar.header("2) Map Columns")
st.sidebar.caption("Students & Teachers are required. Province/Year/Type unlock comparisons.")
cols = list(df.columns)

col_students = st.sidebar.selectbox("Students *", cols, index=None, placeholder="Select...")
col_teachers = st.sidebar.selectbox("Teachers *", cols, index=None, placeholder="Select...")
col_province = st.sidebar.selectbox("Province (optional)", [None] + cols, index=0)
col_year     = st.sidebar.selectbox("Year (optional)",     [None] + cols, index=0)
col_type     = st.sidebar.selectbox("School Type (optional)", [None] + cols, index=0)

if not col_students or not col_teachers:
    st.warning("Select at least Students and Teachers to compute PTR.")
    st.stop()

# Prepare numeric & PTR
work = df.copy()
for c in [col_students, col_teachers]:
    work[c] = pd.to_numeric(work[c], errors="coerce")

work["Students per Teacher"] = np.where(work[col_teachers].fillna(0)==0, np.nan, work[col_students]/work[col_teachers])
work["Teachers per 100 Students"] = np.where(
    work["Students per Teacher"].isna() | (work["Students per Teacher"]==0),
    np.nan, 100.0 / work["Students per Teacher"]
)

# Projection table
keep = [c for c in [col_province, col_year, col_type, col_students, col_teachers,
                    "Students per Teacher", "Teachers per 100 Students"] if c]
clean = work[keep].copy()

# ---------------- Filters ----------------
st.sidebar.header("3) Filters")
filtered = clean.copy()

if col_province:
    prov_vals = sorted(filtered[col_province].dropna().astype(str).unique())
    sel_prov_multi = st.sidebar.multiselect("Filter: Province", prov_vals, default=prov_vals[:min(6, len(prov_vals))])
    if sel_prov_multi:
        filtered = filtered[filtered[col_province].astype(str).isin(sel_prov_multi)]

if col_year:
    year_vals = sorted(filtered[col_year].dropna().astype(str).unique())
    sel_year = st.sidebar.multiselect("Filter: Year", year_vals, default=year_vals[:min(6, len(year_vals))])
    if sel_year:
        filtered = filtered[filtered[col_year].astype(str).isin(sel_year)]

if col_type:
    type_vals = sorted(filtered[col_type].dropna().astype(str).unique())
    sel_type = st.sidebar.multiselect("Filter: School Type", type_vals, default=type_vals[:min(3, len(type_vals))])
    if sel_type:
        filtered = filtered[filtered[col_type].astype(str).isin(sel_type)]

# ---------------- KPIs (overall after filters) ----------------
k1,k2,k3,k4 = st.columns(4)
with k1: st.metric("Mean PTR", f"{filtered['Students per Teacher'].mean():.2f}")
with k2: st.metric("Median PTR", f"{filtered['Students per Teacher'].median():.2f}")
with k3: st.metric("Min PTR", f"{filtered['Students per Teacher'].min():.2f}")
with k4: st.metric("Max PTR", f"{filtered['Students per Teacher'].max():.2f}")

st.markdown("---")

# ---------------- Province Comparison ----------------
st.subheader("🏙️ Province comparison")

if not col_province:
    st.info("Select a Province column in mapping to enable province comparison.")
else:
    prov_all = sorted(filtered[col_province].dropna().astype(str).unique())
    if len(prov_all) < 1:
        st.warning("No provinces available after filters.")
    else:
        cA, cB = st.columns(2)
        with cA:
            p1 = st.selectbox("Primary province", prov_all, index=0, key="p1")
        with cB:
            default_idx = 1 if len(prov_all) > 1 else 0
            p2 = st.selectbox("Comparator province", prov_all, index=default_idx, key="p2")

        def slice_province(df_, p):
            d = df_[df_[col_province].astype(str) == str(p)]
            return d

        d1 = slice_province(filtered, p1)
        d2 = slice_province(filtered, p2)

        c1, c2, c3 = st.columns([1,1,1])

        def kpi_block(df_):
            return {
                "Mean PTR": df_["Students per Teacher"].mean(),
                "Median PTR": df_["Students per Teacher"].median(),
                "Teachers/100 Students": df_["Teachers per 100 Students"].mean()
            }

        m1 = kpi_block(d1)
        m2 = kpi_block(d2)

        with c1:
            st.markdown(f"**{p1}**")
            st.metric("Mean PTR", f"{m1['Mean PTR']:.2f}")
            st.metric("Median PTR", f"{m1['Median PTR']:.2f}")
            st.metric("Avg Teachers/100 Students", f"{m1['Teachers/100 Students']:.2f}")
        with c2:
            st.markdown(f"**{p2}**")
            st.metric("Mean PTR", f"{m2['Mean PTR']:.2f}")
            st.metric("Median PTR", f"{m2['Median PTR']:.2f}")
            st.metric("Avg Teachers/100 Students", f"{m2['Teachers/100 Students']:.2f}")
        with c3:
            st.markdown("**Difference (P1 - P2)**")
            def safe_diff(a,b):
                if pd.isna(a) or pd.isna(b):
                    return np.nan
                return a-b
            def fmt(x): 
                return "—" if pd.isna(x) else f"{x:.2f}"
            st.metric("Δ Mean PTR", fmt(safe_diff(m1["Mean PTR"], m2["Mean PTR"])))
            st.metric("Δ Median PTR", fmt(safe_diff(m1["Median PTR"], m2["Median PTR"])))
            st.metric("Δ Teachers/100", fmt(safe_diff(m1["Teachers/100 Students"], m2["Teachers/100 Students"])))

        st.markdown("#### PTR distribution (selected provinces)")
        dist = filtered.dropna(subset=["Students per Teacher"]).copy()
        dist["__prov"] = dist[col_province].astype(str)
        dist = dist[dist["__prov"].isin([p1, p2])]
        hist = alt.Chart(dist).mark_bar(opacity=0.6).encode(
            alt.X("Students per Teacher:Q", bin=alt.Bin(maxbins=30)),
            alt.Y("count()", title="Count"),
            color="__prov:N"
        ).properties(height=300)
        st.altair_chart(hist, use_container_width=True)

        if col_year:
            st.markdown("#### PTR trend over time")
            t = filtered.copy()
            t["__prov"] = t[col_province].astype(str)
            t["__year"] = t[col_year].astype(str)
            t = t[t["__prov"].isin([p1, p2])]
            agg = t.groupby(["__year","__prov"], as_index=False)["Students per Teacher"].mean()
            line = alt.Chart(agg).mark_line(point=True).encode(
                x=alt.X("__year:N", title="Year"),
                y=alt.Y("Students per Teacher:Q", title="Avg Students per Teacher"),
                color="__prov:N",
                tooltip=["__year","__prov", alt.Tooltip("Students per Teacher:Q", format=".2f")]
            ).properties(height=320)
            st.altair_chart(line, use_container_width=True)

st.markdown("---")

# ---------------- Prediction ----------------
st.subheader("🎯 Teacher prediction from student intake")

if not col_province:
    st.info("Select a Province column to enable province-specific prediction.")
else:
    pcol1, pcol2 = st.columns([1,1])
    with pcol1:
        pred_prov = st.selectbox("Province for prediction", sorted(filtered[col_province].dropna().astype(str).unique()), key="pred_prov")
    with pcol2:
        method = st.selectbox("Prediction method",
                              ["Use mean PTR (Students/Teacher)", "Use median PTR", "Regression: Teachers ~ Students (no intercept)"])

    students_input = st.number_input("Enter number of enrolled students", min_value=0, step=1, value=1000)
    use_filters = st.checkbox("Use current filters (year/type) for this prediction", value=True)

    pred_df = filtered.copy() if use_filters else clean.copy()
    pred_df = pred_df[pred_df[col_province].astype(str) == str(pred_prov)].copy()

    teachers_needed = None
    ptr_used = None
    note = ""

    if pred_df.empty:
        st.warning("No data available for the selected province (with current filters). Adjust filters or use overall data.")
    else:
        if method.startswith("Use mean PTR"):
            ptr_used = pred_df["Students per Teacher"].mean()
            note = "Mean PTR for selected province."
            if pd.notna(ptr_used) and ptr_used > 0:
                teachers_needed = ceil(students_input / ptr_used)
        elif method.startswith("Use median PTR"):
            ptr_used = pred_df["Students per Teacher"].median()
            note = "Median PTR for selected province."
            if pd.notna(ptr_used) and ptr_used > 0:
                teachers_needed = ceil(students_input / ptr_used)
        else:
            x = pd.to_numeric(pred_df[col_students], errors="coerce").fillna(0).values
            y = pd.to_numeric(pred_df[col_teachers], errors="coerce").fillna(0).values
            mask = (x > 0) & (y >= 0)
            x, y = x[mask], y[mask]
            if x.size >= 2:
                denom = (x*x).sum()
                beta = (x*y).sum() / denom if denom != 0 else np.nan
                ptr_used = None
                if pd.notna(beta) and beta > 0:
                    teachers_needed = ceil(beta * students_input)
                    note = f"Regression slope β ≈ {beta:.4f} (Teachers per Student)."
                else:
                    st.warning("Regression failed due to insufficient or invalid data. Try mean/median PTR.")
            else:
                st.warning("Not enough data points for regression. Try mean/median PTR.")

    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        st.metric("Students (input)", f"{students_input:,}")
    with c2:
        if method.startswith("Regression"):
            st.metric("Model", "Teachers ≈ β × Students")
        else:
            st.metric("PTR used (Students/Teacher)", "—" if ptr_used is None or pd.isna(ptr_used) else f"{ptr_used:.2f}")
    with c3:
        st.metric("Teachers needed (ceil)", "—" if teachers_needed is None else f"{teachers_needed:,}")
    if note:
        st.caption(note)

st.markdown("---")

# ---------------- Data table + downloads ----------------
st.subheader("Filtered data")
st.dataframe(filtered, use_container_width=True)

def to_excel_bytes(df_in: pd.DataFrame) -> bytes:
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
        df_in.to_excel(writer, index=False, sheet_name="PTR")
    bio.seek(0)
    return bio.read()

d1, d2 = st.columns(2)
with d1:
    st.download_button("⬇️ Download CSV", filtered.to_csv(index=False).encode("utf-8"),
                       file_name="ptr_filtered.csv", mime="text/csv")
with d2:
    st.download_button("⬇️ Download Excel", to_excel_bytes(filtered),
                       file_name="ptr_filtered.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.caption("Definitions: PTR (Students/Teacher) = Students ÷ Teachers. Teachers per 100 Students = 100 ÷ PTR.")