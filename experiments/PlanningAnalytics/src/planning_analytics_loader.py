"""Load and prepare the anonymized Planning Analytics workbook data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


MONTH_COLUMN_FORMATS = ("%Y-%m-%d", "%Y-%m", "%b-%y", "%b %Y", "%Y/%m/%d")


def _parse_month(value: object) -> pd.Timestamp | None:
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.to_period("M").to_timestamp()
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.notna(parsed):
        return parsed.to_period("M").to_timestamp()
    return None


def _normalise_country(country_token: str) -> str:
    return country_token.replace("_", " ")


def _workbook_metadata(path: Path) -> tuple[str, str]:
    stem = path.stem
    for granularity in ["BN_Leaf_PL_Leaf", "BN_Leaf_PL_Total", "BN_Lvl1_PL_Total"]:
        suffix = f"_{granularity}"
        if stem.endswith(suffix):
            return _normalise_country(stem[: -len(suffix)]), granularity
    raise ValueError(f"Could not infer country/granularity from workbook name: {path.name}")


def _id_columns(columns: list[object]) -> list[object]:
    month_columns = {column for column in columns if _parse_month(column) is not None}
    return [column for column in columns if column not in month_columns]


def _rename_id_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rename_map: dict[object, str] = {}
    for column in frame.columns:
        normalised = str(column).strip().lower().replace(" ", "_")
        if normalised in {"business_unit", "business_units", "bu"}:
            rename_map[column] = "business_unit"
        elif normalised in {"budgetary_nature", "budgetary_natures", "bn"}:
            rename_map[column] = "budgetary_nature"
        elif normalised in {"pl_cost", "pl", "pl_costs"}:
            rename_map[column] = "pl_cost"
    return frame.rename(columns=rename_map)


def load_workbook_long(path: Path) -> pd.DataFrame:
    country, granularity = _workbook_metadata(path)
    raw = pd.read_excel(path, header=None)
    header_row = raw.index[
        raw.apply(
            lambda row: row.astype(str).str.upper().str.contains("BUSINESS_UNIT").any(),
            axis=1,
        )
    ]
    if len(header_row) == 0:
        raise ValueError(f"{path.name} does not contain a BUSINESS_UNIT header row.")
    header_idx = int(header_row[0])
    year_idx = header_idx - 1

    headers = raw.loc[header_idx].tolist()
    if year_idx >= 0:
        years = raw.loc[year_idx].tolist()
    else:
        years = [None] * len(headers)

    columns: list[object] = []
    for position, header in enumerate(headers):
        if position < 3:
            columns.append(str(header).strip())
            continue
        year = years[position]
        month = header
        if pd.isna(year) or pd.isna(month):
            columns.append(header)
            continue
        month_number = int(str(month).upper().replace("M", ""))
        columns.append(pd.Timestamp(year=int(float(year)), month=month_number, day=1))

    wide = raw.iloc[header_idx + 1 :].copy()
    wide.columns = columns
    wide = wide.dropna(how="all").reset_index(drop=True)
    wide = _rename_id_columns(wide)

    required = ["business_unit", "budgetary_nature", "pl_cost"]
    missing = [column for column in required if column not in wide.columns]
    if missing:
        raise ValueError(f"{path.name} is missing expected columns: {missing}")

    id_columns = _id_columns(list(wide.columns))
    value_columns = [column for column in wide.columns if column not in id_columns]
    month_column_map = {
        column: _parse_month(column)
        for column in value_columns
        if _parse_month(column) is not None
    }
    if not month_column_map:
        raise ValueError(f"{path.name} does not contain recognizable monthly columns.")

    frame = wide.melt(
        id_vars=id_columns,
        value_vars=list(month_column_map),
        var_name="date",
        value_name="value",
    )
    frame["date"] = frame["date"].map(month_column_map)
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame["country"] = country
    frame["granularity"] = granularity
    frame["source_file"] = path.name
    frame["source_row"] = frame.groupby("date").cumcount()
    frame["series_id"] = (
        frame["country"].astype(str)
        + "|"
        + frame["granularity"].astype(str)
        + "|"
        + frame["business_unit"].astype(str)
        + "|"
        + frame["budgetary_nature"].astype(str)
        + "|"
        + frame["pl_cost"].astype(str)
    )
    ordered = [
        "source_file",
        "source_row",
        "country",
        "granularity",
        "business_unit",
        "budgetary_nature",
        "pl_cost",
        "date",
        "value",
        "series_id",
    ]
    return frame[ordered].sort_values(["series_id", "date"]).reset_index(drop=True)


def load_all_long(dataset_dir: Path) -> pd.DataFrame:
    paths = sorted(dataset_dir.glob("Country_*_BN_*.xlsx"))
    if not paths:
        raise FileNotFoundError(f"No Planning Analytics workbooks found in {dataset_dir}")
    frames = [load_workbook_long(path) for path in paths]
    return pd.concat(frames, ignore_index=True)


def mark_leading_zeros_as_missing(
    frame: pd.DataFrame,
    *,
    value_column: str = "value",
    output_column: str = "value_leading_zero_na",
) -> pd.DataFrame:
    result = frame.sort_values(["series_id", "date"]).copy()
    result[output_column] = result[value_column].astype("Float64")
    result["is_leading_zero"] = False
    result["is_trailing_zero"] = False
    result["is_edge_zero"] = False

    for _, index in result.groupby("series_id", sort=False).groups.items():
        values = result.loc[index, value_column].fillna(0)
        non_zero = values.ne(0).to_numpy()
        if not non_zero.any():
            continue
        first_non_zero_position = non_zero.nonzero()[0][0]
        leading_index = index[:first_non_zero_position]
        leading_zero_index = leading_index[result.loc[leading_index, value_column].fillna(0).eq(0)]
        result.loc[leading_zero_index, output_column] = pd.NA
        result.loc[leading_zero_index, "is_leading_zero"] = True
        result.loc[leading_zero_index, "is_edge_zero"] = True

    return result
