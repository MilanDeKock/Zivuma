#HBYN
# app.py
# deps: pip install streamlit pandas

from datetime import datetime
import pandas as pd
import streamlit as st


# ---- Page setup ----
st.set_page_config(
    page_title="Zivuma Bulk Assembly Tool",
    page_icon="Zivuma.png",
    layout="centered",
)

# ---- Header row ----
col1, col2 = st.columns([1, 8])  # left column narrow, right wide
with col1:
    st.image("Zivuma.png", width=70)   # logo on the left
with col2:
    st.title("Zivuma Bulk Assembly Tool")  # title next to it


# -------------------------
# Config: column mappings
# -------------------------
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
def read_csv_strip(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df = df.rename(columns=lambda c: str(c).strip())
    return df

def is_availability(df: pd.DataFrame) -> bool:
    needed = {AV["sku"], AV["name"], AV["batch"], AV["onhand"], AV["location"], AV["bin"], AV["expiry"], AV["stock_value"]}
    return needed.issubset(set(df.columns))

def is_bom(df: pd.DataFrame) -> bool:
    return set(REQUIRED_BOM).issubset(set(df.columns))

def prep_availability(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize
    df[AV["onhand"]] = pd.to_numeric(df[AV["onhand"]], errors="coerce").fillna(0)
    df[AV["stock_value"]] = pd.to_numeric(df[AV["stock_value"]], errors="coerce").fillna(0)
    # Treat blanks as NA for key text fields
    for c in [AV["sku"], AV["name"], AV["batch"], AV["location"], AV["bin"]]:
        df[c] = df[c].astype(str).str.strip()
        df.loc[df[c].isin(["", "nan", "None", "NaN"]), c] = pd.NA
    # Ignore rows with blank/NA BatchSerialNumber entirely
    df = df[df[AV["batch"]].notna()].copy()
    # If Bin is NA, keep as NA now; we'll blank it out at export
    return df

def prep_bom(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure all BOM columns present and in order
    for c in BOM_COLS:
        if c not in df.columns:
            df[c] = None
    df = df[BOM_COLS].copy()
    df['Quantity'] = df['Quantity'].astype(int)
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
    """
    Only consider batches with duplicates (count >= 2) after removing blank/NA batches.
    'Clean' means the duplicate rows share the same OnHand and same Location.
    Blanks are treated as NA and excluded from the uniqueness check.
    """
    b, q, loc = AV["batch"], AV["onhand"], AV["location"]

    # Only duplicated batches
    dupmask = av_df.duplicated(subset=[b], keep=False)
    dup = av_df[dupmask].copy()
    if dup.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # For uniqueness, treat blanks as NA
    def nunique_ignore_na(s):
        return s.dropna().nunique()

    def shared_val(s):
        s = s.dropna()
        return s.iloc[0] if s.nunique() == 1 and len(s) > 0 else None

    status = (
        dup.groupby(b)
        .agg(
            qty_unique=(q, nunique_ignore_na),
            loc_unique=(loc, nunique_ignore_na),
            shared_qty=(q, shared_val),
            shared_loc=(loc, shared_val),
            rows=(b, "count"),
        )
        .reset_index()
    )
    status["is_clean"] = (status["qty_unique"] == 1) & (status["loc_unique"] == 1) & status["shared_qty"].notna() & status["shared_loc"].notna()

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
# One upload box (two CSVs)
# -------------------------
uploads = st.file_uploader(
    "Upload two CSV files: Availability Report + Assembly BOM Export",
    type=["csv"],
    accept_multiple_files=True,
    key="multi_csv"
)

if not uploads or len(uploads) < 2:
    st.info("Please upload **two** CSV files (Availability Report + Assembly BOM Export).")
    st.stop()

# Try to auto-detect which is which
av, bom = None, None
dfs = []
for f in uploads:
    try:
        df = read_csv_strip(f)
        dfs.append((f.name, df))
    except Exception as e:
        st.error(f"Could not read {f.name}: {e}")
        st.stop()

for name, df in dfs:
    if is_availability(df) and av is None:
        av = prep_availability(df)
    elif is_bom(df) and bom is None:
        bom = prep_bom(df)

if av is None or bom is None:
    st.error("Could not auto-detect the files. Ensure one CSV contains Availability headers and the other contains BOM headers.")
    # (Optional) Uncomment to view quick previews for debugging:
    # for name, df in dfs:
    #     st.write(name, df.columns.tolist())
    st.stop()

# (Optional PREVIEWS — remove or keep commented to avoid showing)
# with st.expander("Preview Availability (first 100 rows)", expanded=False):
#     st.dataframe(av.head(100), use_container_width=True)
# with st.expander("Preview BOM (first 100 rows)", expanded=False):
#     st.dataframe(bom.head(100), use_container_width=True)

# -------------------------
# Clean vs mismatch
# -------------------------
clean_df, mismatch_df, status = find_clean_batches(av)
if clean_df.empty and mismatch_df.empty:
    st.warning("No duplicate batch numbers found (after excluding blank/NA batches). Nothing to process.")
    st.stop()

# (Optional metrics — comment these out to hide)
# colA, colB = st.columns(2)
# with colA:
#     st.metric("Duplicate batch rows", len(clean_df) + len(mismatch_df))
# with colB:
#     st.metric("Clean batches", int(status["is_clean"].sum()) if not status.empty else 0)

# (Optional mismatch table — comment out to hide)
# if not mismatch_df.empty:
#     with st.expander("Batches needing inspection (quantity/location mismatch)", expanded=False):
#         show = mismatch_df[[AV["batch"], AV["sku"], AV["name"], AV["onhand"], AV["location"]]].copy()
#         st.dataframe(show.sort_values([AV["batch"], AV["sku"]]), use_container_width=True)

if not mismatch_df.empty:
    with st.expander("Batches needing inspection (quantity/location mismatch)", expanded=False):
        show = mismatch_df[[AV["batch"], AV["sku"], AV["name"], AV["onhand"], AV["location"]]].copy()
        st.dataframe(show.sort_values([AV["batch"], AV["sku"]]), use_container_width=True)

if status.empty or status["is_clean"].sum() == 0:
    st.warning("No clean batches to process.")
    st.stop()

# -------------------------
# Per-batch mapping UI (default = not ticked)
# -------------------------
st.subheader("Link Available Batches to a Finished Good (Tick Box & Choose FG SKU)")

# FG options
fg_df = bom[["ProductSKU","ProductName"]].drop_duplicates().sort_values("ProductSKU")
FG_OPTIONS = fg_df["ProductSKU"].tolist()
FG_NAME = dict(zip(fg_df["ProductSKU"], fg_df["ProductName"]))

# Seed mapping (default Use=False)
bcol = AV["batch"]
map_seed = (
    status.loc[status["is_clean"], [bcol, "shared_loc", "shared_qty"]]
    .rename(columns={bcol: "Batch", "shared_loc": "Location", "shared_qty": "FG_Qty"})
    .sort_values("Batch")
)
map_seed["Use"] = False           # <-- default unchecked
map_seed["FG_SKU"] = ""

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

# Keep only selected rows
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
# STRICT mismatch validation: block if any batch↦FG has extra/missing components
# -------------------------
bom_components_by_fg = (
    bom.groupby("ProductSKU")["ComponentSKU"]
       .apply(lambda s: set(s.dropna().astype(str)))
       .to_dict()
)

sel_rows = clean_df[clean_df[bcol].isin(mapping_df["Batch"])].copy()

errors = []
for _, r in mapping_df.iterrows():
    bat = r["Batch"]
    fg_code = r["FG_SKU"]
    batch_components = set(sel_rows.loc[sel_rows[bcol] == bat, AV["sku"]].astype(str).unique())
    bom_set = bom_components_by_fg.get(fg_code, set())
    extra = sorted(batch_components - bom_set)
    missing = sorted(bom_set - batch_components)
    if extra or missing:
        errors.append(f"{bat} ↦ {fg_code}: extra={extra} | missing={missing}")

if errors:
    st.error("Component mismatch — one or more selected batches include SKUs not defined in the selected FG’s BOM.")
    st.stop()

# -------------------------
# Build Stock Adjustment (single combined CSV)
# - Components OUT: NonZero; Quantity = -OnHand; UnitCost = StockValue/OnHand; Bin blanks instead of NaN
# - FG IN: Zero; Quantity = shared OnHand; UnitCost allocated by total component value per batch / FG_Qty
# - ReceivedDate = today (YYYYMMDD)
# -------------------------
TODAY_YYYYMMDD = datetime.now().strftime("%Y%m%d")

# Components OUT
comp = sel_rows[
    [AV["location"], AV["sku"], AV["name"], AV["bin"], AV["batch"], AV["expiry"], AV["onhand"], AV["stock_value"]]
].copy()

# UnitCost = StockValue / OnHand (safe for OnHand==0 -> 0)
comp["UnitCost"] = comp.apply(
    lambda r: (float(r[AV["stock_value"]]) / float(r[AV["onhand"]])) if float(r[AV["onhand"]]) != 0 else 0.0,
    axis=1
).round(4)

comp["Zero/NonZero"] = "NonZero"
comp["ExpiryDate_YYYYMMDD"] = comp[AV["expiry"]].map(yyyymmdd)
comp["Quantity"] = 0
comp["Comments"] = "Auto: Consolidate to FG (per batch mapping)"
comp["ReceivedDate_YYYYMMDD"] = TODAY_YYYYMMDD

# Blank-out Bin if NA to avoid 'nan' in export
comp[AV["bin"]] = comp[AV["bin"]].fillna("")

comps_out = comp.rename(columns={
    AV["location"]: "Location",
    AV["sku"]: "SKU",
    AV["name"]: "Name",
    AV["bin"]: "Bin",
    AV["batch"]: "BatchSerialNumber",
})[ADJ_HEADERS].copy()

# Value per batch to allocate to FG
batch_comp_value = sel_rows.groupby(bcol)[AV["stock_value"]].sum().to_dict()

# FG IN
fg_in_rows = []
FG_NAME = dict(zip(fg_df["ProductSKU"], fg_df["ProductName"]))

for _, r in mapping_df.iterrows():
    bat = r["Batch"]
    fg_code = r["FG_SKU"]
    fg_name = FG_NAME.get(fg_code, "")
    shared_qty = float(r["FG_Qty"])
    shared_loc = r["Location"]

    if shared_qty <= 0:
        continue

    total_value_for_batch = float(batch_comp_value.get(bat, 0.0))
    fg_unit_cost = (total_value_for_batch / shared_qty) if shared_qty != 0 else 0.0
    fg_unit_cost = round(fg_unit_cost, 4)

    fg_in_rows.append({
        "Zero/NonZero": "Zero",
        "Location": shared_loc,
        "SKU": fg_code,
        "Name": fg_name,
        "Bin": "",  # FG line bin left blank (adjust if you need something else)
        "BatchSerialNumber": bat,
        "ExpiryDate_YYYYMMDD": "",
        "Quantity": shared_qty,
        "UnitCost": fg_unit_cost,
        "Comments": f"Auto: Consolidate from {bat}",
        "ReceivedDate_YYYYMMDD": TODAY_YYYYMMDD,
    })

fg_in = pd.DataFrame(fg_in_rows, columns=ADJ_HEADERS)

# Combined export: OUT first, then IN
combined = pd.concat([comps_out, fg_in], ignore_index=True)

st.subheader("Preview of stock adjustment file")
st.dataframe(combined, use_container_width=True)
# -------------------------
# Download
# -------------------------
st.subheader("Download")
ts = datetime.now().strftime("%Y%m%d-%H%M%S")
st.download_button(
    "Download Stock Adjustment CSV (combined)",
    data=combined.to_csv(index=False).encode("utf-8-sig"),
    file_name=f"stock_adjustment_combined_{ts}.csv",
    mime="text/csv",
)

st.caption(
    "Rules: OUT → NonZero (Qty = -OnHand; UnitCost = StockValue/OnHand, Bin blanked). "
    "IN → Zero (Qty = shared OnHand per batch; UnitCost allocated so total IN value equals total OUT value per batch). "
    "ReceivedDate = today (YYYYMMDD). Duplicate-batch checks ignore blank/NA batch numbers."
)
