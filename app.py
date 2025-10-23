# app.py
# deps: pip install streamlit pandas

from datetime import datetime
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Batch → Stock Adjustment (CSV ↔ single output)", layout="wide")
st.title("Batch → Stock Adjustment (CSV uploads → single output)")

# -------------------------
# Config: column mappings
# -------------------------
# Availability CSV expected headers (you gave these; we only require a subset):
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
    # others exist but not needed directly
}

# Assembly BOM CSV expected headers (exact):
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
def read_availability_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df = df.rename(columns=lambda c: str(c).strip())
    needed = [AV["sku"], AV["name"], AV["batch"], AV["onhand"], AV["location"], AV["bin"], AV["expiry"]]
    miss = [c for c in needed if c not in df.columns]
    if miss:
        raise ValueError(f"Availability CSV is missing columns: {miss}")
    df[AV["onhand"]] = pd.to_numeric(df[AV["onhand"]], errors="coerce").fillna(0)
    for c in [AV["sku"], AV["name"], AV["batch"], AV["location"], AV["bin"]]:
        df[c] = df[c].astype(str).str.strip()
    return df

def read_bom_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df = df.rename(columns=lambda c: str(c).strip())
    missing = [c for c in REQUIRED_BOM if c not in df.columns]
    if missing:
        raise ValueError(f"BOM CSV is missing required columns: {missing}")
    for c in BOM_COLS:
        if c not in df.columns:
            df[c] = None
    df = df[BOM_COLS].copy()
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    for c in ["Action","ProductSKU","ProductName","ComponentSKU","ComponentName"]:
        df[c] = df[c].fillna("").astype(str).str.strip()
    if not df["Quantity"].gt(0).all():
        raise ValueError("BOM CSV: all Quantity values must be numeric and > 0.")
    dup = df.duplicated(subset=["ProductSKU","ComponentSKU"], keep=False)
    if dup.any():
        pairs = df.loc[dup, ["ProductSKU","ComponentSKU"]].drop_duplicates().values.tolist()
        raise ValueError(f"BOM CSV has duplicate (ProductSKU, ComponentSKU) pairs: {pairs}")
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
    clean_keys = set(status.loc[status["is_clean"], b])
    clean_df = dup[dup[b].isin(clean_keys)].copy()
    mismatch_df = dup[~dup[b].isin(clean_keys)].copy()
    return clean_df, mismatch_df, status

def yyyymmdd(val):
    if pd.isna(val) or val == "":
        return ""
    try:
        return pd.to_datetime(val).strftime("%Y%m%d")
    except Exception:
        return ""

# -------------------------
# Uploads
# -------------------------
c1, c2 = st.columns(2)
with c1:
    av_file = st.file_uploader("1) Upload Availability Report (CSV)", type=["csv"], key="av_file")
with c2:
    bom_file = st.file_uploader("2) Upload Assembly BOM (CSV)", type=["csv"], key="bom_file")

if not av_file or not bom_file:
    st.info("Upload both CSV files to continue.")
    st.stop()

try:
    av = read_availability_csv(av_file)
except Exception as e:
    st.error(f"Failed to read Availability CSV: {e}")
    st.stop()

try:
    bom = read_bom_csv(bom_file)
except Exception as e:
    st.error(f"Failed to read BOM CSV: {e}")
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

# -------------------------
# Per-batch mapping UI (Batch ____ FG) with checkbox to include
# -------------------------
st.subheader("Link each clean Batch to a Finished Good (one row per batch)")

# Build FG options
fg_df = bom[["ProductSKU","ProductName"]].drop_duplicates().sort_values("ProductSKU")
FG_OPTIONS = fg_df["ProductSKU"].tolist()
FG_NAME = dict(zip(fg_df["ProductSKU"], fg_df["ProductName"]))

# Starter mapping table
bcol = AV["batch"]
map_seed = (
    status.loc[status["is_clean"], [bcol, "shared_loc", "shared_qty"]]
    .rename(columns={bcol: "Batch", "shared_loc": "Location", "shared_qty": "FG_Qty"})
    .sort_values("Batch")
)
map_seed["Use"] = True
map_seed["FG_SKU"] = ""

# Editor on one line: Batch | FG (plus readonly Location, FG_Qty)
mapping_df = st.data_editor(
    map_seed[["Use", "Batch", "FG_SKU", "Location", "FG_Qty"]],
    use_container_width=True,
    hide_index=True,
    key="batch_fg_mapping",
    column_config={
        "Use": st.column_config.CheckboxColumn(help="Tick to include this batch in the adjustment"),
        "Batch": st.column_config.TextColumn(disabled=True),
        "FG_SKU": st.column_config.SelectboxColumn(options=FG_OPTIONS, required=False, help="Pick FG SKU from BOM"),
        "Location": st.column_config.TextColumn(disabled=True),
        "FG_Qty": st.column_config.NumberColumn(disabled=True, help="Auto from shared OnHand"),
    },
)

# Keep only rows marked Use==True
mapping_df = mapping_df[mapping_df["Use"] == True].copy()
if mapping_df.empty:
    st.warning("No batches selected. Tick 'Use' for at least one batch.")
    st.stop()

# Validate FG selections and qty
_missing = mapping_df["FG_SKU"].eq("") | mapping_df["FG_SKU"].isna()
_zeroqty = ~mapping_df["FG_Qty"].astype(float).gt(0)
if _missing.any():
    st.error("Please select an FG SKU for every selected batch.")
    st.stop()
if _zeroqty.any():
    st.error("One or more selected batches have FG_Qty ≤ 0 (not processable).")
    st.stop()

# -------------------------
# Build Stock Adjustment (single combined CSV)
# Components OUT (NonZero; negative qty) + FG IN (Zero; positive qty)
# -------------------------
# Precompute BOM components per FG (for warnings)
bom_components_by_fg = (
    bom.groupby("ProductSKU")["ComponentSKU"]
       .apply(lambda s: set(s.dropna().astype(str)))
       .to_dict()
)

# Rows in Availability for selected batches
sel_rows = clean_df[clean_df[bcol].isin(mapping_df["Batch"])].copy()

# Components OUT
comp = sel_rows[[AV["location"], AV["sku"], AV["name"], AV["bin"], AV["batch"], AV["expiry"], AV["onhand"]]].copy()
comp["Zero/NonZero"] = "NonZero"
comp["ExpiryDate_YYYYMMDD"] = comp[AV["expiry"]].map(yyyymmdd)
comp["Quantity"] = -comp[AV["onhand"]].astype(float)
comp["UnitCost"] = ""
comp["Comments"] = "Auto: Consolidate to FG (see mapping)"
comp["ReceivedDate_YYYYMMDD"] = ""
comps_out = comp.rename(columns={
    AV["location"]: "Location",
    AV["sku"]: "SKU",
    AV["name"]: "Name",
    AV["bin"]: "Bin",
    AV["batch"]: "BatchSerialNumber",
})[ADJ_HEADERS].copy()

# FG IN (one row per mapping row)
fg_in_rows = []
warnings_buf = []
status_idx = status.set_index(bcol)

for _, r in mapping_df.iterrows():
    bat = r["Batch"]
    fg_code = r["FG_SKU"]
    fg_name = FG_NAME.get(fg_code, "")
    shared_qty = float(r["FG_Qty"])
    shared_loc = r["Location"]

    # Per-batch mismatch info (non-blocking)
    batch_components = set(sel_rows.loc[sel_rows[bcol] == bat, AV["sku"]].unique())
    bom_set = bom_components_by_fg.get(fg_code, set())
    extra = sorted(batch_components - bom_set)
    missing = sorted(bom_set - batch_components)
    if extra or missing:
        warnings_buf.append(f"{bat} ↦ {fg_code}: extra {extra} | missing {missing}")

    if shared_qty > 0:
        fg_in_rows.append({
            "Zero/NonZero": "Zero",
            "Location": shared_loc,
            "SKU": fg_code,
            "Name": fg_name,
            "Bin": "",
            "BatchSerialNumber": bat,   # FG batch = component batch
            "ExpiryDate_YYYYMMDD": "",
            "Quantity": shared_qty,
            "UnitCost": "",
            "Comments": f"Auto: Consolidate from {bat}",
            "ReceivedDate_YYYYMMDD": "",
        })

fg_in = pd.DataFrame(fg_in_rows, columns=ADJ_HEADERS)

# Optional warnings
if warnings_buf:
    st.warning("Component mismatches detected:\n- " + "\n- ".join(warnings_buf))

# Combined output (Components OUT first, then FG IN)
combined = pd.concat([comps_out, fg_in], ignore_index=True)

# -------------------------
# Preview & Download (single file)
# -------------------------
st.subheader("Preview (single combined output)")
st.dataframe(combined, use_container_width=True)

ts = datetime.now().strftime("%Y%m%d-%H%M%S")
st.download_button(
    "Download Stock Adjustment CSV (combined)",
    data=combined.to_csv(index=False).encode("utf-8-sig"),
    file_name=f"stock_adjustment_combined_{ts}.csv",
    mime="text/csv",
)

st.caption("Rules: OUT → NonZero (Qty = -OnHand); IN → Zero (Qty = shared OnHand per batch). Location comes from Availability; FG batch inherits the component batch. FG per-batch is chosen on the same line as the batch.")
