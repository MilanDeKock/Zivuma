#HBYN
# deps: pip install streamlit pandas openpyxl
# run: streamlit run app.py

from datetime import datetime
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Batch → Stock Adjustment Helper", layout="wide")
st.title("Batch → Stock Adjustment Helper (Two-file upload)")

# -------------------------
# Config: column mappings
# -------------------------
# Availability (Excel) header row is Excel row 2 -> pandas header=1
AVAIL_HEADER_ROW_INDEX = 1

AV = {
    "category": "Category",
    "sku": "SKU",
    "name": "ProductName",
    "location": "Location",
    "bin": "Bin",
    "batch": "BatchSerialNumber",
    "expiry": "ExpiryDate",
    "stock_value": "StockValue",
    "onhand": "OnHand",
    "available": "Available",
}

# Assembly BOM headers (must match uploaded file)
BOM_COLS = [
    "Action",
    "ProductSKU",
    "ProductName",
    "ComponentSKU",
    "ComponentName",
    "Quantity",
    "WastageQuantity_ForStockComponentOnly",
    "WastagePercent_ForStockComponentOnly",
    "CostPercentage_ForStockComponentOnly",
    "PriceTier_ForServiceComponentOnly",
    "ExpenseAccount_ForServiceComponentOnly",
    "EstimatedUnitCost",
]
REQUIRED_BOM = ["Action","ProductSKU","ProductName","ComponentSKU","ComponentName","Quantity"]

# Stock Adjustment template headers (exact order)
ADJ_HEADERS = [
    "Zero/NonZero",
    "Location",
    "SKU",
    "Name",
    "Bin",
    "BatchSerialNumber",
    "ExpiryDate_YYYYMMDD",
    "Quantity",
    "UnitCost",
    "Comments",
    "ReceivedDate_YYYYMMDD",
]

# -------------------------
# Helpers
# -------------------------
def read_availability(file):
    df = pd.read_excel(file, header=AVAIL_HEADER_ROW_INDEX)
    df = df.rename(columns=lambda c: str(c).strip())
    needed = [AV["sku"], AV["name"], AV["batch"], AV["onhand"], AV["location"], AV["bin"], AV["expiry"]]
    miss = [c for c in needed if c not in df.columns]
    if miss:
        raise ValueError(f"Availability is missing columns: {miss}")
    df[AV["onhand"]] = pd.to_numeric(df[AV["onhand"]], errors="coerce").fillna(0)
    for c in [AV["sku"], AV["name"], AV["batch"], AV["location"], AV["bin"]]:
        df[c] = df[c].astype(str).str.strip()
    return df

def read_bom(file):
    name = file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file, header=0)
    df = df.rename(columns=lambda c: str(c).strip())
    missing = [c for c in REQUIRED_BOM if c not in df.columns]
    if missing:
        raise ValueError(f"BOM is missing required columns: {missing}")
    # Ensure all columns exist + order them
    for c in BOM_COLS:
        if c not in df.columns:
            df[c] = None
    df = df[BOM_COLS].copy()
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    for c in ["Action","ProductSKU","ProductName","ComponentSKU","ComponentName"]:
        df[c] = df[c].fillna("").astype(str).str.strip()
    # basic validation
    if not df["Quantity"].gt(0).all():
        raise ValueError("BOM: all Quantity values must be numeric and > 0.")
    dup = df.duplicated(subset=["ProductSKU","ComponentSKU"], keep=False)
    if dup.any():
        pairs = df.loc[dup, ["ProductSKU","ComponentSKU"]].drop_duplicates().values.tolist()
        raise ValueError(f"BOM has duplicate (ProductSKU, ComponentSKU) pairs: {pairs}")
    return df

def find_clean_batches(av_df: pd.DataFrame):
    b, q, loc = AV["batch"], AV["onhand"], AV["location"]
    dup = av_df[av_df.duplicated(subset=[b], keep=False)].copy()
    if dup.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    status = (
        dup.groupby(b)
        .agg(
            qty_unique=(q, lambda s: s.nunique()),
            loc_unique=(loc, lambda s: s.nunique()),
            shared_qty=(q, lambda s: s.dropna().unique()[0] if s.nunique()==1 else None),
            shared_loc=(loc, lambda s: s.dropna().unique()[0] if s.nunique()==1 else None),
            rows=(b, "count"),
        )
        .reset_index()
    )
    status["is_clean"] = (status["qty_unique"] == 1) & (status["loc_unique"] == 1)
    clean_batches = set(status.loc[status["is_clean"], b])
    clean_df = dup[dup[b].isin(clean_batches)].copy()
    mismatch_df = dup[~dup[b].isin(clean_batches)].copy()
    return clean_df, mismatch_df, status

def yyyymmdd(val):
    if pd.isna(val) or val == "":
        return ""
    try:
        return pd.to_datetime(val).strftime("%Y%m%d")
    except Exception:
        return ""

# -------------------------
# UI: uploads
# -------------------------
col1, col2 = st.columns(2)
with col1:
    av_file = st.file_uploader("1) Upload Availability Report (Excel; header on row 2)", type=["xlsx","xls"])
with col2:
    bom_file = st.file_uploader("2) Upload Assembly BOM (CSV or Excel)", type=["csv","xlsx","xls"])

if not av_file or not bom_file:
    st.info("Upload both files to continue.")
    st.stop()

# Parse inputs
try:
    av = read_availability(av_file)
except Exception as e:
    st.error(f"Failed to read Availability: {e}")
    st.stop()

try:
    bom = read_bom(bom_file)
except Exception as e:
    st.error(f"Failed to read BOM: {e}")
    st.stop()

with st.expander("Preview Availability (first 200 rows)", expanded=False):
    st.dataframe(av.head(200), use_container_width=True)

with st.expander("Preview BOM (first 200 rows)", expanded=False):
    st.dataframe(bom.head(200), use_container_width=True)

# -------------------------
# Clean vs mismatch
# -------------------------
clean_df, mismatch_df, status = find_clean_batches(av)
if clean_df.empty and mismatch_df.empty:
    st.warning("No duplicate batch numbers in Availability. Nothing to process.")
    st.stop()

colA, colB = st.columns(2)
with colA:
    st.metric("Duplicate batch rows", len(clean_df) + len(mismatch_df))
with colB:
    st.metric("Clean batches", int(status["is_clean"].sum()) if not status.empty else 0)

if not mismatch_df.empty:
    with st.expander("Batches needing inspection (qty/location mismatch)", expanded=False):
        show = mismatch_df[[AV["batch"], AV["sku"], AV["name"], AV["onhand"], AV["location"]]].copy()
        st.dataframe(show.sort_values([AV["batch"], AV["sku"]]), use_container_width=True)

if status.empty or status["is_clean"].sum() == 0:
    st.warning("No clean batches to process.")
    st.stop()

# Clean batches table for selection
bcol = AV["batch"]
clean_batches_table = (
    status.loc[status["is_clean"], [bcol, "shared_qty", "shared_loc"]]
    .sort_values(bcol)
    .rename(columns={bcol: "Batch", "shared_qty": "FG_Qty (auto from OnHand)", "shared_loc": "Location"})
)
st.subheader("Select clean batches to convert → Finished Good")
with st.expander("Clean batches", expanded=True):
    st.dataframe(clean_batches_table.reset_index(drop=True), use_container_width=True)

selectable = clean_batches_table["Batch"].tolist()
selected_batches = st.multiselect("Batches", options=selectable)

if not selected_batches:
    st.stop()

# -------------------------
# Finished Good selection (from BOM)
# -------------------------
fg_df = (bom[["ProductSKU","ProductName"]].drop_duplicates().sort_values("ProductSKU"))
fg_label = {r.ProductSKU: f"{r.ProductSKU} — {r.ProductName}".strip(" —") for _, r in fg_df.iterrows()}
fg_code = st.selectbox("Finished Good (from BOM)", options=list(fg_label), format_func=lambda k: fg_label[k])

# Component mismatch info
sel_rows = clean_df[clean_df[AV["batch"]].isin(selected_batches)].copy()
batch_components = set(sel_rows[AV["sku"]].unique())
bom_for_fg = bom[bom["ProductSKU"] == fg_code].copy()
bom_components = set(bom_for_fg["ComponentSKU"].unique())

extra_in_batches = sorted(batch_components - bom_components)
missing_in_batches = sorted(bom_components - batch_components)
if extra_in_batches or missing_in_batches:
    st.warning(
        f"Component mismatch for FG {fg_code}: "
        f"extra in batches {extra_in_batches} | missing from batches {missing_in_batches}"
    )

# -------------------------
# Build Stock Adjustment outputs
# Rules:
# - OUT lines (components): NonZero, Quantity = -OnHand, date formatted
# - IN lines (FG): Zero, Quantity = shared OnHand per selected batch
# - FG BatchSerialNumber defaults to the component batch (no manual input)
# -------------------------
# Components OUT
comp = sel_rows[[AV["location"], AV["sku"], AV["name"], AV["bin"], AV["batch"], AV["expiry"], AV["onhand"]]].copy()
comp["Zero/NonZero"] = "NonZero"
comp["ExpiryDate_YYYYMMDD"] = comp[AV["expiry"]].map(yyyymmdd)
comp["Quantity"] = -comp[AV["onhand"]].astype(float)
comp["UnitCost"] = ""
comp["Comments"] = f"Auto: Consolidate to FG {fg_code}"
comp["ReceivedDate_YYYYMMDD"] = ""

comps_out = comp.rename(columns={
    AV["location"]: "Location",
    AV["sku"]: "SKU",
    AV["name"]: "Name",
    AV["bin"]: "Bin",
    AV["batch"]: "BatchSerialNumber",
})[ADJ_HEADERS].copy()

# FG IN
idx = status.set_index(AV["batch"])
fg_name = bom_for_fg["ProductName"].iloc[0] if not bom_for_fg.empty else ""
fg_rows = []
for bat in selected_batches:
    shared_qty = idx.loc[bat, "shared_qty"]
    shared_loc = idx.loc[bat, "shared_loc"]
    if pd.isna(shared_qty) or float(shared_qty) <= 0:
        continue
    fg_rows.append({
        "Zero/NonZero": "Zero",
        "Location": shared_loc,
        "SKU": fg_code,
        "Name": fg_name,
        "Bin": "",
        "BatchSerialNumber": bat,
        "ExpiryDate_YYYYMMDD": "",
        "Quantity": float(shared_qty),
        "UnitCost": "",
        "Comments": f"Auto: Consolidate from {bat}",
        "ReceivedDate_YYYYMMDD": "",
    })
fg_in = pd.DataFrame(fg_rows, columns=ADJ_HEADERS)

# Optional: combined file toggle
combine = st.checkbox("Also create a single combined CSV (OUT on top, then IN)")

# -------------------------
# Download buttons
# -------------------------
ts = datetime.now().strftime("%Y%m%d-%H%M%S")
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Components OUT (NonZero; negative qty)**")
    st.dataframe(comps_out, use_container_width=True)
    st.download_button(
        "Download Components OUT CSV",
        data=comps_out.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"components_out_{ts}.csv",
        mime="text/csv",
    )
with c2:
    st.markdown("**Finished Good IN (Zero; positive qty)**")
    if fg_in.empty:
        st.warning("No FG IN rows (all shared OnHand were zero or invalid).")
    else:
        st.dataframe(fg_in, use_container_width=True)
        st.download_button(
            "Download FG IN CSV",
            data=fg_in.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"fg_in_{ts}.csv",
            mime="text/csv",
        )

if combine:
    combined = pd.concat([comps_out, fg_in], ignore_index=True)
    st.download_button(
        "Download Combined CSV",
        data=combined.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"stock_adjustments_combined_{ts}.csv",
        mime="text/csv",
    )

st.caption("Template mapping: OUT → NonZero, IN → Zero. Location comes from Availability; FG BatchNumber inherits the component batch.")
