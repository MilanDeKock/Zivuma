#HBYN
# app.py
# deps: pip install streamlit pandas gspread gspread-dataframe openpyxl
# run: streamlit run app.py

from datetime import datetime
import json
import pandas as pd
import streamlit as st
import gspread
from gspread_dataframe import set_with_dataframe

def _secrets_ok():
    try:
        _ = st.secrets["gcp_service_account"]
        _ = st.secrets["BOM_SHEET_ID"]
        return True
    except Exception:
        return False

if not _secrets_ok():
    st.error(
        "Google Sheets secrets not configured. Add `gcp_service_account` and `BOM_SHEET_ID` to Streamlit secrets "
        "(see instructions below). For now, you can upload a BOM CSV/XLSX in the Admin page to use it just for this session."
    )


st.set_page_config(page_title="Batch → Stock Adjustment Helper", layout="wide")

# =========================
# CONFIG (edit if needed)
# =========================
# Availability (Excel) header row is Excel row 2 (=> pandas header=1)
EXCEL_HEADER_ROW_INDEX = 1

# Availability column names (must match your report)
AV_COLS = {
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
    # many others exist but these are the ones we need
}

# Assembly BOM (Google Sheet tab name + required columns)
BOM_SHEET_NAME = "AssemblyBOM"   # name your tab exactly this (or change here)
BOM_LOG_SHEET = "BOM_Changelog"
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
REQUIRED_BOM_COLS = ["Action", "ProductSKU", "ProductName", "ComponentSKU", "ComponentName", "Quantity"]

# Stock Adjustment template headers (exact)
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

# =========================
# GOOGLE SHEETS HELPERS
# =========================
def _gc():
    # Needs secrets:
    # st.secrets["BOM_SHEET_ID"]
    # st.secrets["gcp_service_account"] (service account JSON dict)
    return gspread.service_account_from_dict(st.secrets["gcp_service_account"])

def _open_sheet():
    gc = _gc()
    return gc.open_by_key(st.secrets["BOM_SHEET_ID"])

@st.cache_data(ttl=180)
def load_bom_df():
    """Load Assembly BOM from Google Sheets."""
    sh = _open_sheet()
    try:
        ws = sh.worksheet(BOM_SHEET_NAME)
    except gspread.exceptions.WorksheetNotFound:
        return pd.DataFrame(columns=BOM_COLS)

    data = ws.get_all_records(numericise_ignore=["all"])
    df = pd.DataFrame(data)
    if df.empty:
        return pd.DataFrame(columns=BOM_COLS)

    # Normalize columns and types
    df = df.rename(columns=lambda c: str(c).strip())
    # Ensure all expected columns exist
    for c in BOM_COLS:
        if c not in df.columns:
            df[c] = None
    df = df[BOM_COLS].copy()

    # Quantity numeric
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    # Strings trimmed
    for c in ["Action", "ProductSKU", "ProductName", "ComponentSKU", "ComponentName"]:
        df[c] = df[c].fillna("").astype(str).str.strip()
    return df

def _write_bom_df(new_df: pd.DataFrame):
    """Overwrite AssemblyBOM tab with new_df."""
    sh = _open_sheet()
    try:
        ws_bom = sh.worksheet(BOM_SHEET_NAME)
    except gspread.exceptions.WorksheetNotFound:
        ws_bom = sh.add_worksheet(title=BOM_SHEET_NAME, rows=1, cols=20)
    ws_bom.clear()
    set_with_dataframe(ws_bom, new_df)

def _append_bom_log(row):
    sh = _open_sheet()
    try:
        ws_log = sh.worksheet(BOM_LOG_SHEET)
    except gspread.exceptions.WorksheetNotFound:
        ws_log = sh.add_worksheet(title=BOM_LOG_SHEET, rows=1, cols=12)
        ws_log.append_row(["Timestamp","Uploader","Note","Added","Deleted","Edited","NewRowCount"])
    ws_log.append_row(row)

def _diff_bom(old_df: pd.DataFrame, new_df: pd.DataFrame):
    key = ["ProductSKU", "ComponentSKU"]
    old_k = set(map(tuple, old_df[key].itertuples(index=False, name=None)))
    new_k = set(map(tuple, new_df[key].itertuples(index=False, name=None)))
    added = sorted(list(new_k - old_k))
    deleted = sorted(list(old_k - new_k))
    idx_old = old_df.set_index(key)
    idx_new = new_df.set_index(key)
    edited = []
    for k in (old_k & new_k):
        row_old = idx_old.loc[k][[c for c in BOM_COLS if c not in key]]
        row_new = idx_new.loc[k][[c for c in BOM_COLS if c not in key]]
        if not row_old.equals(row_new):
            edited.append(k)
    return {"added": added, "deleted": deleted, "edited": sorted(edited)}

def _validate_bom(df: pd.DataFrame) -> list[str]:
    errs = []
    for c in REQUIRED_BOM_COLS:
        if c not in df.columns:
            errs.append(f"Missing BOM column: {c}")
    if errs:
        return errs
    # Required values
    missing = df[df["ProductSKU"].eq("") | df["ComponentSKU"].eq("")]
    if not missing.empty:
        errs.append("Some rows have empty ProductSKU or ComponentSKU.")
    # Quantity > 0
    bad = df[~pd.to_numeric(df["Quantity"], errors="coerce").gt(0)]
    if not bad.empty:
        errs.append("BOM Quantity must be numeric and > 0 for all rows.")
    # No duplicate (ProductSKU, ComponentSKU)
    dup = df.duplicated(subset=["ProductSKU", "ComponentSKU"], keep=False)
    if dup.any():
        pairs = df.loc[dup, ["ProductSKU","ComponentSKU"]].drop_duplicates().values.tolist()
        errs.append(f"Duplicate FG+Component pairs: {pairs}")
    return errs

# =========================
# AVAILABILITY HELPERS
# =========================
def read_availability(upload):
    df = pd.read_excel(upload, header=EXCEL_HEADER_ROW_INDEX)
    df = df.rename(columns=lambda c: str(c).strip())
    needed = [AV_COLS["sku"], AV_COLS["name"], AV_COLS["batch"], AV_COLS["onhand"], AV_COLS["location"], AV_COLS["bin"], AV_COLS["expiry"]]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in Availability: {missing}")
    # Normalize
    df[AV_COLS["onhand"]] = pd.to_numeric(df[AV_COLS["onhand"]], errors="coerce").fillna(0)
    for c in [AV_COLS["sku"], AV_COLS["name"], AV_COLS["batch"], AV_COLS["location"], AV_COLS["bin"]]:
        df[c] = df[c].astype(str).str.strip()
    return df

def find_clean_batches(df: pd.DataFrame):
    b, q, loc = AV_COLS["batch"], AV_COLS["onhand"], AV_COLS["location"]
    dup = df[df.duplicated(subset=[b], keep=False)].copy()
    if dup.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    status = (
        dup.groupby(b)
        .agg(
            qty_unique=(q, lambda s: s.nunique()),
            loc_unique=(loc, lambda s: s.nunique()),
            shared_qty=(q, lambda s: s.dropna().unique()[0] if s.nunique()==1 else None),
            shared_loc=(loc, lambda s: s.dropna().unique()[0] if s.nunique()==1 else None),
            rows=("SKU", "count")
        )
        .reset_index()
    )
    status["is_clean"] = (status["qty_unique"] == 1) & (status["loc_unique"] == 1)
    clean_batches = set(status.loc[status["is_clean"], b])
    clean_df = dup[dup[b].isin(clean_batches)].copy()
    mismatch_df = dup[~dup[b].isin(clean_batches)].copy()
    return clean_df, mismatch_df, status

def yyyymmdd(val):
    # Convert Excel/str/ts to 'YYYYMMDD' or ''
    if pd.isna(val) or val == "":
        return ""
    try:
        dt = pd.to_datetime(val)
        return dt.strftime("%Y%m%d")
    except Exception:
        return ""

# =========================
# UI
# =========================
st.title("Batch → Stock Adjustment Helper")

mode = st.sidebar.radio("Mode", ["Process Batches", "Admin: Replace BOM"])

if mode == "Admin: Replace BOM":
    st.subheader("Admin • Replace Assembly BOM")
    cur_bom = load_bom_df()
    st.caption(f"Current BOM rows: {len(cur_bom)} (tab: {BOM_SHEET_NAME})")
    with st.expander("Preview current BOM (first 200 rows)", expanded=False):
        st.dataframe(cur_bom.head(200), use_container_width=True)

    upload_bom = st.file_uploader("Upload new BOM file (CSV or Excel)", type=["csv","xlsx","xls"], key="bom_file")
    uploader = st.text_input("Your name (for audit log)", value="")
    note = st.text_input("Change note (optional)", value="")
    passphrase = st.text_input("Admin passphrase", type="password", value="")
    validate_btn = st.button("Validate & Preview replacement")

    if validate_btn:
        if not upload_bom:
            st.error("Please upload a BOM file.")
            st.stop()
        # Parse
        name = upload_bom.name.lower()
        if name.endswith(".csv"):
            new_bom = pd.read_csv(upload_bom)
        else:
            new_bom = pd.read_excel(upload_bom, header=0)
        new_bom = new_bom.rename(columns=lambda c: str(c).strip())

        # Ensure all BOM_COLS exist
        missing = [c for c in REQUIRED_BOM_COLS if c not in new_bom.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
            st.stop()

        # Normalize to BOM_COLS order
        for c in BOM_COLS:
            if c not in new_bom.columns:
                new_bom[c] = None
        new_bom = new_bom[BOM_COLS].copy()
        # Types
        new_bom["Quantity"] = pd.to_numeric(new_bom["Quantity"], errors="coerce")
        for c in ["Action","ProductSKU","ProductName","ComponentSKU","ComponentName"]:
            new_bom[c] = new_bom[c].fillna("").astype(str).str.strip()

        errs = _validate_bom(new_bom)
        if errs:
            st.error("Validation failed:\n- " + "\n- ".join(errs))
            st.stop()

        st.success("Validation passed.")
        with st.expander("Preview new BOM (first 200 rows)", expanded=True):
            st.dataframe(new_bom.head(200), use_container_width=True)

        # Diff
        summary = _diff_bom(cur_bom, new_bom)
        st.info(f"Diff — Added: {len(summary['added'])}, Deleted: {len(summary['deleted'])}, Edited: {len(summary['edited'])}")
        with st.expander("Diff details", expanded=False):
            st.write("Added keys:", summary["added"])
            st.write("Deleted keys:", summary["deleted"])
            st.write("Edited keys:", summary["edited"])

        replace_btn = st.button("Replace BOM with uploaded file", type="primary",
                                disabled=not uploader or not passphrase)
        if replace_btn:
            if passphrase != st.secrets.get("EDITOR_PASS", ""):
                st.error("Invalid passphrase.")
            else:
                try:
                    _write_bom_df(new_bom)
                    _append_bom_log([
                        datetime.now().isoformat(timespec="seconds"),
                        uploader, note,
                        json.dumps(summary["added"]),
                        json.dumps(summary["deleted"]),
                        json.dumps(summary["edited"]),
                        len(new_bom),
                    ])
                    load_bom_df.clear()  # bust cache
                    st.success("BOM replaced. Processing page will use the new data.")
                except Exception as e:
                    st.error(f"Replace failed: {e}")

else:
    st.subheader("Process Batches → Build Stock Adjustments")

    # Load BOM (for FG dropdown + checks)
    bom_df = load_bom_df()
    if bom_df.empty:
        st.warning(f"No BOM found in Google Sheet tab '{BOM_SHEET_NAME}'. Add it via Admin first.")
    else:
        # Build FG list from BOM
        fg_df = (bom_df[["ProductSKU","ProductName"]]
                 .drop_duplicates()
                 .sort_values("ProductSKU"))
        fg_label = {r.ProductSKU: f"{r.ProductSKU} — {r.ProductName}".strip(" —") for _, r in fg_df.iterrows()}

    # Upload availability
    up = st.file_uploader("Upload Availability Report (Excel; header on row 2)", type=["xlsx","xls"])
    if not up:
        st.stop()

    try:
        av = read_availability(up)
    except Exception as e:
        st.error(f"Could not read Availability: {e}")
        st.stop()

    with st.expander("Preview Availability (first 300 rows)", expanded=False):
        st.dataframe(av.head(300), use_container_width=True)

    # Find duplicate batches and split into clean vs mismatch
    clean_df, mismatch_df, status = find_clean_batches(av)
    if clean_df.empty and mismatch_df.empty:
        st.info("No duplicate batch numbers found in Availability.")
        st.stop()

    colA, colB = st.columns(2)
    with colA:
        st.metric("Duplicate batch rows", len(clean_df) + len(mismatch_df))
    with colB:
        st.metric("Clean batches", int(status["is_clean"].sum()) if not status.empty else 0)

    if not mismatch_df.empty:
        with st.expander("Batches needing inspection (qty/location mismatch)", expanded=False):
            show = mismatch_df[[AV_COLS["batch"], AV_COLS["sku"], AV_COLS["name"], AV_COLS["onhand"], AV_COLS["location"]]].copy()
            st.dataframe(show.sort_values([AV_COLS["batch"], AV_COLS["sku"]]), use_container_width=True)

    if clean_df.empty:
        st.warning("No clean batches to process.")
        st.stop()

    # Clean batches list
    bcol, qcol, lcol = AV_COLS["batch"], AV_COLS["onhand"], AV_COLS["location"]
    clean_batches = (status.loc[status["is_clean"], [bcol, "shared_qty", "shared_loc"]]
                     .sort_values(bcol)
                     .rename(columns={bcol: "Batch", "shared_qty":"FG_Qty", "shared_loc":"Location"}))

    st.subheader("Select batches to convert → FG")
    with st.expander("Clean batches (auto FG Qty = shared OnHand)", expanded=True):
        st.dataframe(clean_batches.reset_index(drop=True), use_container_width=True)

    selectable = clean_batches["Batch"].tolist()
    selected_batches = st.multiselect("Batches", options=selectable)

    if not selected_batches:
        st.stop()

    # Choose FG (from BOM)
    if bom_df.empty:
        st.error("No BOM loaded. Cannot choose Finished Good.")
        st.stop()
    fg_code = st.selectbox("Finished Good (from BOM)", options=list(fg_label), format_func=lambda k: fg_label[k])

    # Mismatch check vs BOM (informational)
    sel_rows = clean_df[clean_df[bcol].isin(selected_batches)].copy()
    batch_components = set(sel_rows[AV_COLS["sku"]].unique())

    bom_for_fg = bom_df[bom_df["ProductSKU"] == fg_code].copy()
    bom_components = set(bom_for_fg["ComponentSKU"].unique())

    extra_in_batches = sorted(batch_components - bom_components)
    missing_in_batches = sorted(bom_components - batch_components)
    if extra_in_batches or missing_in_batches:
        st.warning(
            f"Component mismatch for FG {fg_code}: "
            f"extra in batches {extra_in_batches} | missing from batches {missing_in_batches}"
        )

    # Build outputs according to your Stock Adjustment template
    # Components OUT (one row per component line in Availability for selected batches)
    comp_rows = sel_rows[[AV_COLS["location"], AV_COLS["sku"], AV_COLS["name"], AV_COLS["bin"], AV_COLS["batch"], AV_COLS["expiry"], AV_COLS["onhand"]]].copy()
    comp_rows["Zero/NonZero"] = "NonZero"  # OUT = NonZero (per your rule)
    comp_rows["ExpiryDate_YYYYMMDD"] = comp_rows[AV_COLS["expiry"]].map(yyyymmdd)
    comp_rows["Quantity"] = -comp_rows[AV_COLS["onhand"]].astype(float)  # negative
    comp_rows["UnitCost"] = ""  # leave blank unless you want to compute
    comp_rows["Comments"] = f"Auto: Consolidate to FG {fg_code}"
    comp_rows["ReceivedDate_YYYYMMDD"] = ""

    comps_out = comp_rows.rename(columns={
        AV_COLS["location"]: "Location",
        AV_COLS["sku"]: "SKU",
        AV_COLS["name"]: "Name",
        AV_COLS["bin"]: "Bin",
        AV_COLS["batch"]: "BatchSerialNumber",
    })[ADJ_HEADERS].copy()

    # FG IN (one row per selected batch) with FG Qty = shared OnHand for that batch
    # Pull the shared qty/location from 'status'
    idx = status.set_index(bcol)
    fg_in_rows = []
    fg_name = bom_for_fg["ProductName"].iloc[0] if not bom_for_fg.empty else ""
    for bat in selected_batches:
        shared_qty = idx.loc[bat, "shared_qty"]
        shared_loc = idx.loc[bat, "shared_loc"]
        if pd.isna(shared_qty) or float(shared_qty) <= 0:
            continue
        fg_in_rows.append({
            "Zero/NonZero": "Zero",       # IN = Zero (per your rule)
            "Location": shared_loc,
            "SKU": fg_code,
            "Name": fg_name,
            "Bin": "",
            "BatchSerialNumber": bat,      # defaults to component batch
            "ExpiryDate_YYYYMMDD": "",
            "Quantity": float(shared_qty), # positive
            "UnitCost": "",
            "Comments": f"Auto: Consolidate from {bat}",
            "ReceivedDate_YYYYMMDD": "",
        })
    fg_in = pd.DataFrame(fg_in_rows, columns=ADJ_HEADERS)

    # Output previews + downloads
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    comps_name = f"components_out_{ts}.csv"
    fg_name_csv = f"fg_in_{ts}.csv"

    st.subheader("Preview & Export")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Components OUT (NonZero; negative qty)**")
        st.dataframe(comps_out, use_container_width=True)
        st.download_button("Download Components OUT CSV", data=comps_out.to_csv(index=False).encode("utf-8-sig"),
                           file_name=comps_name, mime="text/csv")
    with c2:
        st.markdown("**Finished Good IN (Zero; positive qty)**")
        if fg_in.empty:
            st.warning("No FG IN rows (all shared OnHand were zero or invalid).")
        else:
            st.dataframe(fg_in, use_container_width=True)
            st.download_button("Download FG IN CSV", data=fg_in.to_csv(index=False).encode("utf-8-sig"),
                               file_name=fg_name_csv, mime="text/csv")

    st.caption("Note: If you want a single combined file, we can output one CSV with both OUT and IN rows stacked.")

