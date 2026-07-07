#!/usr/bin/env python3
"""
Combine an external H3N2 HI-titre table with WHO-HI extractor outputs.

This script is designed to complement who_hi_extractor_fasta_auto.py.
It reads a previously published HI-titre CSV table, links virus/reference names
with a FASTA file, calculates the same fixed-scale HI score, and combines the
result with the scored output from who_hi_extractor_fasta_auto.py.

Main use case:
    python who_hi_conbiner.py \
      --external-hi prior_H3N2_HI_table.csv \
      --who-score-csv NH2021_score.csv SH2022_score.csv \
      --fasta H3N2_sequences.fasta \
      --out combined_HI_score.csv \
      --external-out prior_H3N2_scored.csv \
      --log combined_HI_score_log.csv

Note:
    This version uses external-table-style underscore-separated uppercase keys
    for FASTA matching. For example, A_TRIESTE_25C_2007 from the external HI
    table and A/Trieste/25c/2007 from FASTA are matched using the shared key
    A_TRIESTE_25C_2007.

    When a FASTA match is found, the final output strain name adopts the
    FASTA-derived display name, preserving FASTA casing such as 25c.

    In external HI tables, strain names may have suffixes after the year,
    such as A/WISCONSIN/67/2005#ISOLATE2. These suffixes are removed before
    name normalization and FASTA matching.

    If the external HI table does not provide virus_collection_date, the script
    fills it from a matched FASTA header collection date when available, e.g.
    A/Victoria/3/1975|...|1975-01-01|HA|...
"""

from __future__ import annotations

from pathlib import Path
import argparse
import glob
import random
import re
import sys
from datetime import datetime, date
from typing import Iterable, Optional

import numpy as np
import pandas as pd


# -----------------------------
# Basic utilities
# -----------------------------

def cell_to_str(x) -> str:
    if x is None:
        return ""
    if pd.isna(x):
        return ""
    if isinstance(x, str):
        return re.sub(r"\s+", " ", x).strip()
    if isinstance(x, (datetime, date)):
        return x.strftime("%Y-%m-%d")
    return str(x).strip()


def expand_two_digit_year(yy: int) -> int:
    """
    Expand 2-digit years using the influenza-oriented 1930-2029 rule.

    Examples:
        72 -> 1972
        05 -> 2005
    """
    return 1900 + yy if 30 <= yy <= 99 else 2000 + yy


def strip_post_year_suffix_from_strain_name(name: str | None) -> Optional[str]:
    """
    Remove annotations appended after the strain-name year without mistaking
    year-like internal strain-name fields for the collection year.

    Rule:
        Slash-form strain names are interpreted as A/location/.../isolate/year.
        The year field is the rightmost slash-separated token that starts with
        a valid influenza year. This supports locations that themselves contain
        slashes, such as A/LYON/CHU/2182/1999, while avoiding truncation at
        year-like internal fields such as A/Panama/2007/1999.

        Anything after that year field is removed.

        Underscore-form names are handled analogously by taking the rightmost
        token that starts with a valid influenza year as the year field. This
        allows locations with underscores, e.g. A_HONG_KONG_2671_2019.

    Examples:
        A/LYON/CHU/2182/1999           -> A/LYON/CHU/2182/1999
        A/LYON/CHU/2182/1999@AA1A      -> A/LYON/CHU/2182/1999
        A/Panama/2007/1999             -> A/Panama/2007/1999
        A/Panama/2007/1999@AA1A        -> A/Panama/2007/1999
        A/Wisconsin/67/2005ISOLATE2    -> A/Wisconsin/67/2005
        A/Wisconsin/67/2005@AA1A       -> A/Wisconsin/67/2005
        A/Wisconsin/67/2005/1          -> A/Wisconsin/67/2005
        A/Trieste/25c/2007             -> A/Trieste/25c/2007
        A/St. Petersburg/1/2009        -> A/St Petersburg/1/2009
        A/O'Hare/1/2009                -> A/OHare/1/2009
        IVR-238 (A/Victoria/4897/2022@AA1A)
                                           -> IVR-238 (A/Victoria/4897/2022)
    """
    x = cell_to_str(name)
    if not x:
        return None

    x = x.replace("（", "(").replace("）", ")").strip()
    x = x.replace(".", "").replace("'", "")

    year_prefix_re = re.compile(r"^(19[3-9]\d|20[0-2]\d|\d{2})(.*)$")

    def clean_slash_strain(s: str) -> str:
        parts = s.split("/")
        if len(parts) < 4 or parts[0].upper() not in {"A", "B"}:
            return s

        # Use the rightmost slash-separated token that starts with a valid year.
        # This supports names with additional location fields, e.g.
        # A/LYON/CHU/2182/1999, while avoiding the internal year-like isolate in
        # A/Panama/2007/1999.
        best_i = None
        best_year = None
        for i in range(len(parts) - 1, 0, -1):
            m = year_prefix_re.match(parts[i].rstrip(")"))
            if m:
                best_i = i
                best_year = m.group(1)
                break

        if best_i is None:
            return s

        return "/".join(parts[:best_i] + [best_year])

    def clean_underscore_strain(s: str) -> str:
        parts = [p for p in s.split("_") if p != ""]
        if len(parts) < 4 or parts[0].upper() not in {"A", "B"}:
            return s

        # Underscore-form names may contain underscores in the location
        # component. Therefore, use the rightmost token that starts with a valid
        # year as the year field, then remove everything after it.
        best_i = None
        best_year = None
        for i in range(len(parts) - 1, 0, -1):
            m = year_prefix_re.match(parts[i].rstrip(")"))
            if m:
                best_i = i
                best_year = m.group(1)
                break

        if best_i is None:
            return s

        return "_".join(parts[:best_i] + [best_year])

    def clean_plain_strain(s: str) -> str:
        s = s.strip()
        if re.match(r"^[AB]/", s, flags=re.I):
            return clean_slash_strain(s)
        if re.match(r"^[AB]_", s, flags=re.I):
            return clean_underscore_strain(s)
        return s

    # Clean a strain name inside reassortant/vaccine notation first.
    # Example: IVR-238 (A/Victoria/4897/2022@AA1A)
    def replace_parenthetical(m: re.Match) -> str:
        return f"({clean_plain_strain(m.group(1))})"

    x = re.sub(r"\((A[/_][^)]+)\)", replace_parenthetical, x, flags=re.I)

    # If an annotation follows a cleaned parenthetical strain name, remove it.
    # Example: IVR-238 (A/Victoria/4897/2022)@note
    x = re.sub(r"^(.+\(A[/_][^)]+\)).+$", r"\1", x, flags=re.I)

    return clean_plain_strain(x)

def expand_terminal_year_in_strain_name(name: str | None) -> Optional[str]:
    """
    Expand only the rightmost slash/underscore-separated year token.

    Supports both slash-form and external-table underscore-form names. Internal
    year-like tokens are preserved and never interpreted as the collection year.

    Examples:
        A/Kamata/85/1987       -> A/Kamata/85/1987
        A/Tokyo/1/51           -> A/Tokyo/1/1951
        A/Memphis/102/72       -> A/Memphis/102/1972
        A/LYON/CHU/20/2000    -> A/LYON/CHU/20/2000
        A/LYON/CHU/20/00      -> A/LYON/CHU/20/2000
        A_MEMPHIS_102_72       -> A_MEMPHIS_102_1972
    """
    x = strip_post_year_suffix_from_strain_name(name)
    if not x:
        return None

    x = x.replace("（", "(").replace("）", ")")

    def expand_last_token(s: str) -> str:
        s = s.strip()
        close = ")" if s.endswith(")") else ""
        core = s[:-1] if close else s

        sep = "/" if re.match(r"^[AB]/", core, flags=re.I) else "_" if re.match(r"^[AB]_", core, flags=re.I) else None
        if sep is None:
            # Non-standard fallback: only expand a truly terminal /YY or _YY token.
            m = re.search(r"([/_])(\d{2})$", core)
            if m:
                return core[:m.start()] + f"{m.group(1)}{expand_two_digit_year(int(m.group(2)))}" + close
            return s

        parts = core.split(sep) if sep == "/" else [p for p in core.split("_") if p != ""]
        if len(parts) >= 4 and re.fullmatch(r"\d{2}", parts[-1]):
            parts[-1] = f"{expand_two_digit_year(int(parts[-1]))}"
            core = sep.join(parts)
        return core + close

    # First expand strain names inside reassortant/vaccine parentheses.
    x = re.sub(
        r"\((A[/_][^)]+)\)",
        lambda m: f"({expand_last_token(m.group(1))})",
        x,
        flags=re.I,
    )

    # Then expand if the whole value itself is a strain name.
    return expand_last_token(x)


def extract_year_from_strain_name(name: str | None) -> Optional[int]:
    """
    Extract collection year from a strain name.

    Prefer a terminal 4-digit year. If unavailable, use a terminal 2-digit year.
    Supports both slash-form and external-table underscore-form names.
    """
    x = strip_post_year_suffix_from_strain_name(name)
    if not x:
        return None

    x = x.replace("（", "(").replace("）", ")")

    m4 = re.search(r"[/_](19[3-9]\d|20[0-2]\d)(?:\)?$)", x)
    if m4:
        return int(m4.group(1))

    m2 = re.search(r"[/_](\d{2})(?:\)?$)", x)
    if m2:
        return expand_two_digit_year(int(m2.group(1)))

    return None


def normalize_location_for_match(location: str | None) -> str:
    """
    Normalize the location component for matching.

    Examples:
        Trieste   -> TRIESTE
        HONG_KONG -> HONGKONG
        Hong Kong -> HONGKONG
    """
    s = cell_to_str(location).upper()
    s = re.sub(r"[_\s\-]+", "", s)
    s = re.sub(r"[^A-Z0-9]", "", s)
    return s


def normalize_isolate_for_match(isolate: str | None) -> str:
    """
    Normalize the isolate component for matching.

    Examples:
        25c -> 25C
        25C -> 25C
    """
    s = cell_to_str(isolate).upper()
    s = re.sub(r"[_\s\-]+", "", s)
    s = re.sub(r"[^A-Z0-9]", "", s)
    return s


def parse_strain_components_for_match(name: str | None) -> Optional[tuple[str, str, str, str]]:
    """
    Parse a strain name into A/B, location, isolate, year components.

    Supports:
        A/Trieste/25c/2007
        A_TRIESTE_25C_2007
        A_HONG_KONG_2671_2019

    Returns normalized matching components:
        (A, TRIESTE, 25C, 2007)
    """
    x = expand_terminal_year_in_strain_name(name)
    if not x:
        return None

    x = x.replace("（", "(").replace("）", ")").strip()

    # If this is a reassortant/vaccine string containing a strain in parentheses,
    # canonicalize the parent strain for matching.
    m = re.search(r"\((A[/_][^)]+)\)", x, flags=re.I)
    if m:
        inner = parse_strain_components_for_match(m.group(1))
        if inner:
            return inner

    # Slash-form FASTA/display strain name: A/Trieste/25c/2007
    if re.match(r"^[AB]/", x, flags=re.I):
        parts = x.split("/")
        if len(parts) >= 4:
            virus_type = parts[0].upper()
            year = parts[-1].rstrip(")")
            isolate = parts[-2]
            location = "".join(parts[1:-2])
            if re.fullmatch(r"19[3-9]\d|20[0-2]\d", year):
                return (
                    virus_type,
                    normalize_location_for_match(location),
                    normalize_isolate_for_match(isolate),
                    year,
                )

    # External-table underscore-form strain name: A_TRIESTE_25C_2007.
    # If a location itself contains underscores, e.g. A_HONG_KONG_2671_2019,
    # all middle tokens are joined as the location.
    if re.match(r"^[AB]_", x, flags=re.I):
        parts = [p for p in x.split("_") if p != ""]
        if len(parts) >= 4:
            virus_type = parts[0].upper()
            year = parts[-1].rstrip(")")
            isolate = parts[-2]
            location = "".join(parts[1:-2])
            if re.fullmatch(r"19[3-9]\d|20[0-2]\d", year):
                return (
                    virus_type,
                    normalize_location_for_match(location),
                    normalize_isolate_for_match(isolate),
                    year,
                )

    return None


def format_location_token(token: str) -> str:
    """
    Format a location or isolate token for fallback output.

    Exact original casing is recovered from FASTA when possible. This function
    is only used when no FASTA match is available.
    """
    t = cell_to_str(token)
    if not t:
        return t

    # Normalize numeric / alphanumeric isolate parts.
    # Examples: 25c -> 25C, 25C -> 25C, 102 -> 102.
    if re.fullmatch(r"\d+[A-Za-z]", t):
        return t[:-1] + t[-1].upper()

    if re.fullmatch(r"\d+", t):
        return t

    # Keep other mixed-case names as-is, e.g. Trieste.
    if any(ch.islower() for ch in t):
        return t

    # A few common joined locations where underscores/spaces are often lost.
    joined_location_map = {
        "HONGKONG": "Hong_Kong",
        "NEWYORK": "New_York",
        "NEWJERSEY": "New_Jersey",
        "SOUTHAFRICA": "South_Africa",
        "STPETERSBURG": "St_Petersburg",
        "SANFRANCISCO": "San_Francisco",
        "LOSANGELES": "Los_Angeles",
        "NORTHCAROLINA": "North_Carolina",
        "SOUTHCAROLINA": "South_Carolina",
    }
    map_key = re.sub(r"[_\s\-]+", "", t.upper())
    if map_key in joined_location_map:
        return joined_location_map[map_key]

    # Fallback: title case.
    return t[:1].upper() + t[1:].lower()


def output_format_strain_name(name: str | None) -> Optional[str]:
    """
    Convert strain names to an output-friendly slash-form format.

    If the input is already slash-form, preserve all internal slash-separated
    fields. This avoids corrupting names such as A/LYON/CHU/20/2000. If the
    input is underscore-form, convert it to a readable slash-form fallback.

    Examples:
        A/WISCONSIN/67/2005      -> A/WISCONSIN/67/2005
        A/LYON/CHU/20/2000      -> A/LYON/CHU/20/2000
        A_MEMPHIS_102_72         -> A/Memphis/102/1972
        A_TRIESTE_25C_2007       -> A/Trieste/25C/2007
    """
    x = expand_terminal_year_in_strain_name(name)
    if not x:
        return None

    # Slash-form names may contain extra location/sub-location fields. Preserve
    # their slash structure rather than reconstructing as A/location/isolate/year.
    if re.match(r"^[AB]/", x, flags=re.I):
        x = re.sub(r"\s+", "_", x.strip())
        parts = x.split("/")
        parts[0] = parts[0].upper()
        return "/".join(parts)

    # For underscore-form names, keep the previous readable slash-form fallback.
    comp = parse_strain_components_for_match(x)
    if comp:
        virus_type, location_key, isolate_key, year = comp
        return f"{virus_type}/{format_location_token(location_key)}/{format_location_token(isolate_key)}/{year}"

    x = re.sub(r"\s+", "_", x.strip())
    return x


def fasta_display_strain_name(name: str | None) -> Optional[str]:
    """
    Display name used when a strain is matched to FASTA.

    Matching uses canonical_strain_name(), but the final output should adopt the
    FASTA strain name rather than reformatting it from the external table.
    Therefore this function preserves FASTA casing, including isolate suffix
    letters such as 25c.

    Examples:
        A/Trieste/25c/2007 -> A/Trieste/25c/2007
        A/Memphis/102/72   -> A/Memphis/102/1972
        A/Hong Kong/1/2019 -> A/Hong_Kong/1/2019
    """
    x = expand_terminal_year_in_strain_name(name)
    if not x:
        return None

    # FASTA is normally slash-form. Preserve casing and only normalize spaces.
    if re.match(r"^[AB]/", x, flags=re.I):
        x = re.sub(r"\s+", "_", x.strip())
        parts = x.split("/")
        if parts:
            parts[0] = parts[0].upper()
        return "/".join(parts)

    # For rare underscore-form FASTA names, return a readable slash-form fallback.
    return output_format_strain_name(x)


def canonical_strain_name(name: str | None) -> str:
    """
    Build the external-table-style matching key used for FASTA lookup.

    The key is uppercase and underscore-separated:
        A_TRIESTE_25C_2007

    Therefore these names match:
        prior_H3N2_HI_table.csv: A_TRIESTE_25C_2007
        FASTA header strain name: A/Trieste/25c/2007

    Post-year suffixes are removed and only the terminal 2-digit year is expanded.
    """
    comp = parse_strain_components_for_match(name)
    if comp:
        virus_type, location_key, isolate_key, year = comp
        return f"{virus_type}_{location_key}_{isolate_key}_{year}"

    # Fallback for non-standard names.
    x = expand_terminal_year_in_strain_name(name)
    if not x:
        return ""
    x = x.upper()
    x = re.sub(r"[/\s\-]+", "_", x)
    x = re.sub(r"[^A-Z0-9_()]", "", x)
    x = re.sub(r"_+", "_", x).strip("_")
    return x


def candidate_match_names(name: str | None) -> list[str]:
    """
    Candidate names/aliases for FASTA lookup.
    """
    raw = strip_post_year_suffix_from_strain_name(name)
    out = []

    if raw:
        out.append(raw)

        # Reassortant / vaccine-strain notation:
        # IVR-238_(A/Victoria/4897/2022) -> also try A/Victoria/4897/2022
        m = re.search(r"\((A[/_][^)]+)\)", raw, flags=re.I)
        if m:
            out.append(m.group(1))

    seen = set()
    candidates = []
    for x in out:
        # Keep raw candidate because canonical_strain_name() can now parse both
        # slash-form and underscore-form names.
        for sx in [x, output_format_strain_name(x)]:
            if sx and sx not in seen:
                seen.add(sx)
                candidates.append(sx)

    return candidates

# -----------------------------
# FASTA handling
# -----------------------------

def looks_like_influenza_strain_name(x: str | None) -> bool:
    """
    Return True if a FASTA header field looks like an influenza strain name.

    Examples:
        A/Kamata/85/1987      -> True
        A/Memphis/102/72      -> True
        B/Victoria/2/1987     -> True
        A_/_H3N2              -> False
        EPI_ISL_21051         -> False
        HA                    -> False
    """
    s = cell_to_str(x)
    if not s:
        return False
    if not re.match(r"^[AB]/", s, flags=re.I):
        return False
    return extract_year_from_strain_name(s) is not None


def extract_fasta_strain_name(header: str | None) -> Optional[str]:
    """
    Extract strain name from H1N1-style or H3N2-style FASTA headers.

    H1N1-style:
        >EPI275|HA|A/Kamata/85/1987|EPI_ISL_107|A_/_H1N1
        -> A/Kamata/85/1987

    H3N2-style:
        >A/Memphis/102/72|EPI_ISL_21051|A_/_H3N2|||unassigned|1972-01-01|HA|EPI118986
        -> A/Memphis/102/72
    """
    if not header:
        return None

    parts = [cell_to_str(p) for p in header.split("|")]

    for p in parts:
        if looks_like_influenza_strain_name(p):
            return p

    if len(parts) >= 3 and parts[2]:
        return parts[2]

    if len(parts) >= 1 and parts[0]:
        return parts[0].split()[0]

    return None


def extract_fasta_collection_date(header: str | None) -> Optional[str]:
    """
    Extract collection date from a FASTA header when present.

    H3N2-style example:
        >A/Victoria/3/1975|EPI_ISL_1096|A_/_H3N2|||3C.2|1975-01-01|HA|EPI118970
        -> 1975-01-01

    The function searches pipe-delimited fields for a valid ISO-like date.
    If no full date exists, it returns None rather than guessing from the strain name.
    """
    if not header:
        return None

    parts = [cell_to_str(p) for p in header.split("|")]

    # Prefer a field that is exactly an ISO date.
    for p in parts:
        if re.fullmatch(r"(?:19[3-9]\d|20[0-2]\d)-\d{1,2}-\d{1,2}", p):
            try:
                return pd.to_datetime(p, errors="raise").date().isoformat()
            except Exception:
                pass

    # Fallback: find an ISO-like date embedded in a field.
    for p in parts:
        m = re.search(r"\b(19[3-9]\d|20[0-2]\d)-\d{1,2}-\d{1,2}\b", p)
        if m:
            try:
                return pd.to_datetime(m.group(0), errors="raise").date().isoformat()
            except Exception:
                pass

    return None


def clean_sequence(seq_chunks: list[str]) -> str:
    """Join FASTA sequence lines and remove gap characters."""
    return (
        "".join(seq_chunks)
        .replace(" ", "")
        .replace("\n", "")
        .replace("\r", "")
        .replace("-", "")
        .strip()
    )


def read_fasta_index(fasta_path: str | Path | None) -> dict:
    """
    Read FASTA and build canonical-name indexes.

    Returns:
        {
            "seq_by_canon": canonical strain name -> sequence,
            "display_by_canon": canonical strain name -> FASTA-derived display strain name,
            "collection_date_by_canon": canonical strain name -> collection date from FASTA header,
            "n_records": number of FASTA records,
            "n_records_with_collection_date": number of records with a header collection date,
            "n_duplicate_canonical_names": number of duplicate canonical-name records,
        }

    If multiple FASTA records map to the same canonical strain name, select a
    sequence with the fewest X residues. If multiple records tie for the lowest
    X count, randomly choose one of the tied records.
    """
    if fasta_path is None:
        return {
            "seq_by_canon": {},
            "display_by_canon": {},
            "collection_date_by_canon": {},
            "n_records": 0,
            "n_records_with_collection_date": 0,
            "n_duplicate_canonical_names": 0,
            "n_canonical_names_with_duplicate_records": 0,
            "n_canonical_names_with_top_x_ties": 0,
        }

    fasta_path = Path(fasta_path)
    records_by_canon: dict[str, list[dict[str, Optional[str]]]] = {}
    n_records = 0
    n_records_with_collection_date = 0

    current_name = None
    current_collection_date = None
    current_seq: list[str] = []

    def flush():
        nonlocal current_name, current_collection_date, current_seq
        nonlocal n_records, n_records_with_collection_date

        if current_name and current_seq:
            # FASTA-derived display name is preserved for final output.
            # Matching still uses a fully uppercased canonical key.
            display_name = fasta_display_strain_name(current_name)
            canon = canonical_strain_name(current_name)
            seq = clean_sequence(current_seq)

            n_records += 1
            if current_collection_date:
                n_records_with_collection_date += 1

            if canon and seq:
                records_by_canon.setdefault(canon, []).append(
                    {
                        "seq": seq,
                        "display_name": display_name,
                        "collection_date": current_collection_date,
                    }
                )

        current_name = None
        current_collection_date = None
        current_seq = []

    with fasta_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")

            if line.startswith(">"):
                flush()
                header = line[1:].strip()
                current_name = extract_fasta_strain_name(header)
                current_collection_date = extract_fasta_collection_date(header)
            else:
                current_seq.append(line.strip())

    flush()

    seq_by_canon: dict[str, str] = {}
    display_by_canon: dict[str, Optional[str]] = {}
    collection_date_by_canon: dict[str, Optional[str]] = {}
    n_duplicates = 0
    n_duplicate_canonical_names = 0
    n_top_x_ties = 0

    for canon, records in records_by_canon.items():
        if len(records) > 1:
            n_duplicate_canonical_names += 1
            n_duplicates += len(records) - 1

        for rec in records:
            rec["x_count"] = rec["seq"].upper().count("X") if rec.get("seq") else 0

        min_x = min(rec["x_count"] for rec in records)
        best_records = [rec for rec in records if rec["x_count"] == min_x]
        if len(best_records) > 1:
            n_top_x_ties += 1

        chosen = random.choice(best_records)
        seq_by_canon[canon] = chosen["seq"]
        display_by_canon[canon] = chosen["display_name"]
        collection_date_by_canon[canon] = chosen["collection_date"]

    return {
        "seq_by_canon": seq_by_canon,
        "display_by_canon": display_by_canon,
        "collection_date_by_canon": collection_date_by_canon,
        "n_records": n_records,
        "n_records_with_collection_date": n_records_with_collection_date,
        "n_duplicate_canonical_names": n_duplicates,
        "n_canonical_names_with_duplicate_records": n_duplicate_canonical_names,
        "n_canonical_names_with_top_x_ties": n_top_x_ties,
    }

def lookup_name_sequence_and_collection_date(
    name: str | None,
    fasta_index: dict,
) -> tuple[Optional[str], Optional[str], Optional[str], bool]:
    """
    Return restored_name, sequence, collection_date_from_fasta, matched_to_fasta.

    If no FASTA match is found, restored_name is the best-effort output-format name.
    """
    seq_by_canon = fasta_index.get("seq_by_canon", {})
    display_by_canon = fasta_index.get("display_by_canon", {})
    collection_date_by_canon = fasta_index.get("collection_date_by_canon", {})

    for cand in candidate_match_names(name):
        canon = canonical_strain_name(cand)
        if canon in seq_by_canon:
            return (
                display_by_canon.get(canon, output_format_strain_name(cand)),
                seq_by_canon[canon],
                collection_date_by_canon.get(canon),
                True,
            )

    return output_format_strain_name(name), None, None, False


# Backward-compatible alias for older calls, if any downstream code imports it.
def lookup_name_and_sequence(name: str | None, fasta_index: dict) -> tuple[Optional[str], Optional[str], bool]:
    restored_name, seq, _collection_date, matched = lookup_name_sequence_and_collection_date(name, fasta_index)
    return restored_name, seq, matched


# -----------------------------
# HI table parsing
# -----------------------------

def parse_hi_value(value):
    """
    Parse HI titre values.

    Censor rule follows who_hi_extractor_fasta_auto.py:
        640      -> HI_titre=640,  censor=0
        '<40'    -> HI_titre=40,   censor=1
        '<'      -> HI_titre=40,   censor=1
        '>5120'  -> HI_titre=5120, censor=0
        '>=5120' -> HI_titre=5120, censor=0
    """
    raw = cell_to_str(value)
    if raw == "":
        return None, None, None

    s = raw.replace(",", "").replace("≤", "<=").replace("≥", ">=").strip()
    censor = 1 if "<" in s else 0

    if s in {"<", "<="}:
        return 40, 1, raw

    m = re.search(r"(\d+(?:\.\d+)?)", s)
    if not m:
        return None, censor if censor else None, raw

    num = float(m.group(1))
    if num.is_integer():
        num = int(num)

    return num, censor, raw


def normalize_passage_type(x) -> str:
    """
    Normalize passage labels.

    The external table may contain mixed passage histories such as EGG/CELL,
    CELL/EGG, E3/SIAT, or P1/SIAT/E3. This function preserves the biological
    order after normalization, so EGG->CELL histories become EGG|CELL and
    CELL->EGG histories become CELL|EGG.
    """
    s = cell_to_str(x).upper()
    if not s or s in {"NA", "N/A", "N.D.", "ND", "UNKNOWN", "UNK", "NONE", "-", "--", "?"}:
        return "UNKNOWN"

    s = s.replace(" ", "")
    parts = re.split(r"[|;/,+]+", s)
    normalized = []

    for p in parts:
        if not p:
            continue
        if re.search(r"EGG|\bE\d+\b|\bEX\b", p):
            normalized.append("EGG")
        elif re.search(r"CELL|MDCK|SIAT|C\d|P\d", p):
            normalized.append("CELL")
        elif p in {"EGG", "CELL"}:
            normalized.append(p)
        else:
            # Keep unknown but non-empty passage labels rather than silently dropping them.
            normalized.append(p)

    if not normalized:
        return "UNKNOWN"

    # Preserve first-occurrence order.
    # Examples:
    #   E3/SIAT    -> EGG|CELL
    #   P1/SIAT/E3 -> CELL|EGG
    #   E3/E4      -> EGG
    unique = []
    for p in normalized:
        if p not in unique:
            unique.append(p)

    return "|".join(unique)


def parse_iso_date(x) -> Optional[str]:
    s = cell_to_str(x)
    if not s:
        return None
    try:
        return pd.to_datetime(s, errors="raise").date().isoformat()
    except Exception:
        return None


def derive_season_from_date(date_value: str | None) -> Optional[str]:
    """
    Derive influenza season using the updated naming convention:
        SHYYYY: YYYY-02-01 to YYYY-08-31
        NHYYYY: (YYYY-1)-09-01 to YYYY-01-31

    Examples:
        2014-09-01 -> NH2015
        2015-01-31 -> NH2015
        2015-02-01 -> SH2015
        2015-09-01 -> NH2016
    """
    if not date_value:
        return None

    try:
        d = pd.to_datetime(date_value, errors="raise").date()
    except Exception:
        return None

    if 2 <= d.month <= 8:
        return f"SH{d.year}"
    if 9 <= d.month <= 12:
        return f"NH{d.year + 1}"
    if d.month == 1:
        return f"NH{d.year}"
    return None


def fill_collection_date_from_name(name: str | None) -> tuple[Optional[str], bool]:
    year = extract_year_from_strain_name(name)
    if year is None:
        return None, False
    return f"{year:04d}-01-01", True


def make_strain_passage_id(strain, passage_type) -> Optional[str]:
    strain_s = cell_to_str(strain)
    passage_s = cell_to_str(passage_type)
    if not strain_s or not passage_s:
        return None
    return f"{strain_s}_{passage_s}"


def is_nonempty_sequence(x) -> bool:
    if pd.isna(x):
        return False
    return bool(str(x).strip())


def count_x_in_sequence(x) -> int:
    if pd.isna(x):
        return 0
    return str(x).upper().count("X")


def build_log(metrics: dict) -> pd.DataFrame:
    return pd.DataFrame([{"metric": key, "value": value} for key, value in metrics.items()])


# -----------------------------
# External H3N2 table processing
# -----------------------------

def read_external_hi_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)

    required = {"virus", "reference", "virus_passage", "reference_passage", "date", "titre"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"External HI table is missing required column(s): {missing}")

    return df


def normalize_external_hi_table(
    external_df: pd.DataFrame,
    fasta_index: dict,
    source_name: str,
) -> tuple[pd.DataFrame, dict]:
    """
    Normalize the external HI table and attach sequences from FASTA.
    """
    records = []
    metrics = {}

    metrics["external_input_rows"] = int(len(external_df))

    terminal_2digit_name_count = 0
    fasta_virus_match_count = 0
    fasta_reference_match_count = 0

    for i, row in external_df.iterrows():
        virus_raw = cell_to_str(row.get("virus"))
        reference_raw = cell_to_str(row.get("reference"))
        virus_clean_for_match = strip_post_year_suffix_from_strain_name(virus_raw)
        reference_clean_for_match = strip_post_year_suffix_from_strain_name(reference_raw)

        if re.search(r"/\d{2}$", cell_to_str(virus_clean_for_match)):
            terminal_2digit_name_count += 1
        if re.search(r"/\d{2}$", cell_to_str(reference_clean_for_match)):
            terminal_2digit_name_count += 1

        virus_name, virus_seq, virus_fasta_collection_date, virus_matched = lookup_name_sequence_and_collection_date(
            virus_clean_for_match,
            fasta_index,
        )
        reference_name, reference_seq, reference_fasta_collection_date, reference_matched = lookup_name_sequence_and_collection_date(
            reference_clean_for_match,
            fasta_index,
        )

        fasta_virus_match_count += int(virus_matched)
        fasta_reference_match_count += int(reference_matched)

        hi_titre, censor, hi_raw = parse_hi_value(row.get("titre"))

        # Preserve invalid external assay-date strings instead of converting them to missing.
        # Valid dates are normalized to YYYY-MM-DD; invalid non-empty strings such as
        # "1990-aaa" are kept as-is in the date column, with season set to None.
        date_raw = cell_to_str(row.get("date"))
        date_iso = parse_iso_date(date_raw)
        date_value = date_iso if date_iso is not None else (date_raw if date_raw else None)
        date_is_invalid_preserved = bool(date_raw and date_iso is None)

        # External H3N2 HI tables often lack virus_collection_date.
        # Priority:
        #   1. virus_collection_date column in the external HI table, if present
        #   2. matched FASTA header collection date, e.g. 1975-01-01
        #   3. fallback to YYYY-01-01 inferred from the virus strain name
        external_collection_date = parse_iso_date(row.get("virus_collection_date"))
        name_collection_date, inferred_from_name = fill_collection_date_from_name(virus_name)

        if external_collection_date:
            virus_collection_date = external_collection_date
            inferred_collection_date = False
            virus_collection_date_source = "external_table"
        elif virus_fasta_collection_date:
            virus_collection_date = virus_fasta_collection_date
            inferred_collection_date = False
            virus_collection_date_source = "fasta_header"
        else:
            virus_collection_date = name_collection_date
            inferred_collection_date = inferred_from_name
            virus_collection_date_source = "strain_name_year" if name_collection_date else "missing"

        records.append(
            {
                "source_file": source_name,
                "sheet_name": pd.NA,
                "data_source": "external_h3n2_table",
                "season": derive_season_from_date(date_iso),
                "date": date_value,
                "date_raw": date_raw if date_raw else pd.NA,
                "date_is_invalid_preserved": date_is_invalid_preserved,
                "date_is_season_fallback": False,
                "virus_raw": virus_raw,
                "reference_raw": reference_raw,
                "virus_clean_for_match": virus_clean_for_match,
                "reference_clean_for_match": reference_clean_for_match,
                "virus": virus_name,
                "reference": reference_name,
                "reference_from_header": pd.NA,
                "reference_from_row": reference_name,
                "reference_match_score": pd.NA,
                "virus_passage": normalize_passage_type(row.get("virus_passage")),
                "reference_passage": normalize_passage_type(row.get("reference_passage")),
                "virus_passage_type": normalize_passage_type(row.get("virus_passage")),
                "reference_passage_type": normalize_passage_type(row.get("reference_passage")),
                "virus_passage_raw": row.get("virus_passage"),
                "reference_passage_raw": row.get("reference_passage"),
                "virus_collection_date": virus_collection_date,
                "virus_collection_date_source": virus_collection_date_source,
                "virus_collection_date_is_inferred_from_name": inferred_collection_date,
                "virus_fasta_collection_date": virus_fasta_collection_date,
                "reference_fasta_collection_date": reference_fasta_collection_date,
                "HI_titre": hi_titre,
                "titre": row.get("titre"),
                "censor": censor if censor is not None else 0,
                "HI_raw": hi_raw,
                "virus_seq": virus_seq,
                "reference_seq": reference_seq,
                "virus_fasta_matched": virus_matched,
                "reference_fasta_matched": reference_matched,
                "antiserum": row.get("antiserum", pd.NA),
                "RBC": row.get("RBC", pd.NA),
                "pair": row.get("pair", pd.NA),
                "pair_strict": row.get("pair_strict", pd.NA),
                "external_row": int(i),
            }
        )

    out = pd.DataFrame(records)

    metrics["external_terminal_2digit_year_names_seen"] = int(terminal_2digit_name_count)
    metrics["external_rows_with_virus_post_year_suffix_removed"] = int(
        (out["virus_raw"].astype(str) != out["virus_clean_for_match"].astype(str)).sum()
    ) if not out.empty else 0
    metrics["external_rows_with_reference_post_year_suffix_removed"] = int(
        (out["reference_raw"].astype(str) != out["reference_clean_for_match"].astype(str)).sum()
    ) if not out.empty else 0
    metrics["external_virus_names_matched_to_fasta_rows"] = int(fasta_virus_match_count)
    metrics["external_reference_names_matched_to_fasta_rows"] = int(fasta_reference_match_count)
    metrics["external_rows_missing_virus_fasta_match"] = int(len(out) - fasta_virus_match_count)
    metrics["external_rows_missing_reference_fasta_match"] = int(len(out) - fasta_reference_match_count)
    metrics["external_rows_with_invalid_date_preserved"] = int(
        out["date_is_invalid_preserved"].sum()
    ) if not out.empty and "date_is_invalid_preserved" in out.columns else 0
    metrics["external_rows_with_virus_collection_date_from_external_table"] = int(
        (out["virus_collection_date_source"] == "external_table").sum()
    ) if not out.empty else 0
    metrics["external_rows_with_virus_collection_date_from_fasta_header"] = int(
        (out["virus_collection_date_source"] == "fasta_header").sum()
    ) if not out.empty else 0
    metrics["external_rows_with_virus_collection_date_from_strain_name_year"] = int(
        (out["virus_collection_date_source"] == "strain_name_year").sum()
    ) if not out.empty else 0
    metrics["external_rows_missing_virus_collection_date"] = int(
        (out["virus_collection_date_source"] == "missing").sum()
    ) if not out.empty else 0
    metrics["external_unique_virus_names_output"] = int(out["virus"].nunique(dropna=True)) if not out.empty else 0
    metrics["external_unique_reference_names_output"] = int(out["reference"].nunique(dropna=True)) if not out.empty else 0

    return out, metrics


def build_missing_fasta_match_table(external_long: pd.DataFrame) -> pd.DataFrame:
    """
    Build a unique strain-name table for external HI records that failed FASTA matching.

    The row-level metrics external_rows_missing_virus_fasta_match and
    external_rows_missing_reference_fasta_match count HI table rows. This table is
    the corresponding unique strain-name list, separated by whether the missing
    match occurred on the virus side or the reference side.
    """
    if external_long is None or external_long.empty:
        return pd.DataFrame(
            columns=[
                "match_side",
                "raw_name",
                "normalized_name",
                "canonical_name",
                "n_rows",
                "first_external_row",
                "example_dates",
            ]
        )

    records = []

    specs = [
        {
            "match_side": "virus",
            "matched_col": "virus_fasta_matched",
            "raw_col": "virus_raw",
            "name_col": "virus",
        },
        {
            "match_side": "reference",
            "matched_col": "reference_fasta_matched",
            "raw_col": "reference_raw",
            "name_col": "reference",
        },
    ]

    for spec in specs:
        matched_col = spec["matched_col"]
        raw_col = spec["raw_col"]
        name_col = spec["name_col"]

        if matched_col not in external_long.columns or name_col not in external_long.columns:
            continue

        missing = external_long[external_long[matched_col] == False].copy()  # noqa: E712

        if missing.empty:
            continue

        if raw_col not in missing.columns:
            missing[raw_col] = missing[name_col]

        missing["canonical_name_for_missing_report"] = missing[name_col].apply(canonical_strain_name)

        group_cols = [raw_col, name_col, "canonical_name_for_missing_report"]

        for (raw_name, normalized_name, canonical_name), g in missing.groupby(group_cols, dropna=False):
            dates = sorted(d for d in g.get("date", pd.Series(dtype="object")).dropna().astype(str).unique())
            example_dates = ";".join(dates[:5])

            records.append(
                {
                    "match_side": spec["match_side"],
                    "raw_name": raw_name,
                    "normalized_name": normalized_name,
                    "canonical_name": canonical_name,
                    "n_rows": int(len(g)),
                    "first_external_row": int(g["external_row"].min()) if "external_row" in g.columns else pd.NA,
                    "example_dates": example_dates,
                }
            )

    out = pd.DataFrame(records)

    if out.empty:
        return pd.DataFrame(
            columns=[
                "match_side",
                "raw_name",
                "normalized_name",
                "canonical_name",
                "n_rows",
                "first_external_row",
                "example_dates",
            ]
        )

    return out.sort_values(
        ["match_side", "n_rows", "normalized_name"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def build_scored_filtered_dataset(
    df: pd.DataFrame,
    fixed_score_scale: float = 8.0,
    source_prefix: str = "external",
    titre_log_diff_violate_cutoff: float = -1.0,
    titre_log_diff_violate_rate_threshold: float = 0.5,
    remove_target_rows_linked_to_low_self_reference_groups: bool = True,
    low_self_titre_log_margin: float = 2.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate fixed-scale HI score and apply the same sequence filters as
    who_hi_extractor_fasta_auto.py.
    """
    metrics = {}

    if df is None or df.empty:
        metrics[f"{source_prefix}_input_rows_for_scoring"] = 0
        metrics[f"{source_prefix}_final_scored_rows"] = 0
        return pd.DataFrame(), build_log(metrics)

    data = df.copy()
    metrics[f"{source_prefix}_input_rows_for_scoring"] = int(len(data))

    before = len(data)
    data = data.drop_duplicates().reset_index(drop=True)
    metrics[f"{source_prefix}_dropped_exact_duplicate_rows"] = int(before - len(data))

    data["HI_titre"] = pd.to_numeric(data.get("HI_titre"), errors="coerce")
    before = len(data)
    data = data[data["HI_titre"].notna() & (data["HI_titre"] > 0)].copy()
    metrics[f"{source_prefix}_dropped_missing_or_invalid_HI_titre_rows"] = int(before - len(data))

    virus_seq_present = data["virus_seq"].apply(is_nonempty_sequence)
    reference_seq_present = data["reference_seq"].apply(is_nonempty_sequence)
    metrics[f"{source_prefix}_rows_missing_virus_seq_before_sequence_filter"] = int((~virus_seq_present).sum())
    metrics[f"{source_prefix}_rows_missing_reference_seq_before_sequence_filter"] = int((~reference_seq_present).sum())
    metrics[f"{source_prefix}_rows_missing_either_seq_before_sequence_filter"] = int((~(virus_seq_present & reference_seq_present)).sum())

    before = len(data)
    data = data[virus_seq_present & reference_seq_present].copy()
    metrics[f"{source_prefix}_dropped_missing_sequence_rows"] = int(before - len(data))

    data["virus_seq_X_count"] = data["virus_seq"].apply(count_x_in_sequence)
    data["reference_seq_X_count"] = data["reference_seq"].apply(count_x_in_sequence)
    either_x_bad = (data["virus_seq_X_count"] >= 4) | (data["reference_seq_X_count"] >= 4)

    metrics[f"{source_prefix}_rows_with_either_seq_X_count_ge_4"] = int(either_x_bad.sum())
    before = len(data)
    data = data[~either_x_bad].copy()
    metrics[f"{source_prefix}_dropped_X_count_ge_4_rows"] = int(before - len(data))

    data["titre_log"] = np.log2(data["HI_titre"] / 10)
    data["virus_strain_passage"] = [
        make_strain_passage_id(v, p)
        for v, p in zip(data["virus"], data["virus_passage"])
    ]
    data["reference_strain_passage"] = [
        make_strain_passage_id(v, p)
        for v, p in zip(data["reference"], data["reference_passage"])
    ]

    strict_self = data[data["virus_strain_passage"] == data["reference_strain_passage"]].copy()
    metrics[f"{source_prefix}_strict_self_titre_rows_after_filters"] = int(len(strict_self))
    metrics[f"{source_prefix}_strict_self_titre_groups_after_filters"] = (
        int(strict_self[["date", "reference_strain_passage"]].drop_duplicates().shape[0])
        if not strict_self.empty
        else 0
    )

    self_mean = (
        strict_self.groupby(["date", "reference_strain_passage"], dropna=False)["titre_log"]
        .mean()
        .reset_index()
        .rename(columns={"titre_log": "self_mean_titre_log"})
    )

    data = data.merge(
        self_mean,
        on=["date", "reference_strain_passage"],
        how="left",
    )

    data["titre_log_diff"] = data["self_mean_titre_log"] - data["titre_log"]

    before = len(data)
    data = data[data["titre_log_diff"].notna()].copy()
    metrics[f"{source_prefix}_dropped_rows_without_strict_self_titre"] = int(before - len(data))

    # Group-level QC: remove an entire date/reference/reference_passage group when
    # too many rows have titre_log_diff <= the violation cutoff. This catches
    # serum/date groups where many heterologous titres are higher than the self
    # titre, before the row-level titre_log_diff filter is applied.
    metrics[f"{source_prefix}_titre_log_diff_violate_cutoff"] = float(titre_log_diff_violate_cutoff)
    metrics[f"{source_prefix}_titre_log_diff_violate_rate_threshold"] = float(
        titre_log_diff_violate_rate_threshold
    )
    metrics[f"{source_prefix}_remove_target_rows_linked_to_low_self_reference_groups"] = bool(
        remove_target_rows_linked_to_low_self_reference_groups
    )
    metrics[f"{source_prefix}_low_self_titre_log_margin"] = float(low_self_titre_log_margin)

    if not 0 <= titre_log_diff_violate_rate_threshold <= 1:
        raise ValueError(
            "titre_log_diff_violate_rate_threshold must be between 0 and 1. "
            f"Got: {titre_log_diff_violate_rate_threshold}"
        )

    reference_passage_group_col = (
        "reference_passage_type" if "reference_passage_type" in data.columns else "reference_passage"
    )
    group_cols = ["date", "reference", reference_passage_group_col]
    metrics[f"{source_prefix}_titre_log_diff_violate_group_cols"] = ";".join(group_cols)

    if data.empty:
        data["titre_log_diff_group_n"] = pd.Series(dtype="int64")
        data["titre_log_diff_violate_n"] = pd.Series(dtype="int64")
        data["titre_log_diff_violate_rate"] = pd.Series(dtype="float64")
        data["titre_log_diff_viorate_rate"] = pd.Series(dtype="float64")
        data["titre_log_diff_group_removed_by_violate_rate"] = pd.Series(dtype="bool")
        data["date_median_self_mean_titre_log"] = pd.Series(dtype="float64")
        data["low_self_reference_group"] = pd.Series(dtype="bool")
        data["target_row_removed_by_low_self_reference_group"] = pd.Series(dtype="bool")
        metrics[f"{source_prefix}_titre_log_diff_violate_groups_before_filter"] = 0
        metrics[f"{source_prefix}_titre_log_diff_groups_removed_by_violate_rate"] = 0
        metrics[f"{source_prefix}_low_self_reference_groups_flagged_for_target_filter"] = 0
        metrics[f"{source_prefix}_rows_removed_by_titre_log_diff_violate_rate"] = 0
        metrics[f"{source_prefix}_rows_removed_by_low_self_reference_target_filter"] = 0
    else:
        data["titre_log_diff_is_violation"] = data["titre_log_diff"] <= titre_log_diff_violate_cutoff
        group_stats = (
            data.groupby(group_cols, dropna=False)
            .agg(
                titre_log_diff_group_n=("titre_log_diff", "size"),
                titre_log_diff_violate_n=("titre_log_diff_is_violation", "sum"),
                reference_group_self_mean_titre_log=("self_mean_titre_log", "first"),
            )
            .reset_index()
        )
        group_stats["titre_log_diff_violate_rate"] = (
            group_stats["titre_log_diff_violate_n"] / group_stats["titre_log_diff_group_n"]
        )
        group_stats["titre_log_diff_viorate_rate"] = group_stats["titre_log_diff_violate_rate"]
        group_stats["titre_log_diff_group_removed_by_violate_rate"] = (
            group_stats["titre_log_diff_violate_rate"] > titre_log_diff_violate_rate_threshold
        )

        metrics[f"{source_prefix}_titre_log_diff_violate_groups_before_filter"] = int(len(group_stats))
        metrics[f"{source_prefix}_titre_log_diff_groups_removed_by_violate_rate"] = int(
            group_stats["titre_log_diff_group_removed_by_violate_rate"].sum()
        )

        # Target-side QC for low-reactive homologous viruses:
        # If a bad reference group also has a self titre that is much lower than
        # the same-date median self titre, the problem is likely caused by the
        # virus/target strain itself rather than only by the serum column. In that
        # case, remove rows where the same strain appears as the target virus in
        # the same date and passage context. This is enabled by default.
        date_medians = (
            group_stats[["date", "reference_group_self_mean_titre_log"]]
            .dropna(subset=["reference_group_self_mean_titre_log"])
            .groupby("date", dropna=False)["reference_group_self_mean_titre_log"]
            .median()
            .reset_index()
            .rename(columns={"reference_group_self_mean_titre_log": "date_median_self_mean_titre_log"})
        )
        group_stats = group_stats.merge(date_medians, on="date", how="left")
        group_stats["low_self_reference_group"] = (
            group_stats["titre_log_diff_group_removed_by_violate_rate"]
            & group_stats["reference_group_self_mean_titre_log"].notna()
            & group_stats["date_median_self_mean_titre_log"].notna()
            & (
                group_stats["reference_group_self_mean_titre_log"]
                <= group_stats["date_median_self_mean_titre_log"] - low_self_titre_log_margin
            )
        )

        if not remove_target_rows_linked_to_low_self_reference_groups:
            group_stats["low_self_reference_group"] = False

        metrics[f"{source_prefix}_low_self_reference_groups_flagged_for_target_filter"] = int(
            group_stats["low_self_reference_group"].sum()
        )

        data = data.merge(group_stats, on=group_cols, how="left")

        virus_passage_group_col = (
            "virus_passage_type" if "virus_passage_type" in data.columns else "virus_passage"
        )
        low_self_targets = group_stats[group_stats["low_self_reference_group"]][
            ["date", "reference", reference_passage_group_col]
        ].rename(
            columns={
                "reference": "virus",
                reference_passage_group_col: virus_passage_group_col,
            }
        )
        low_self_targets = low_self_targets.drop_duplicates()
        low_self_targets["target_row_removed_by_low_self_reference_group"] = True

        data = data.merge(
            low_self_targets,
            on=["date", "virus", virus_passage_group_col],
            how="left",
        )
        data["target_row_removed_by_low_self_reference_group"] = data[
            "target_row_removed_by_low_self_reference_group"
        ].map(lambda x: bool(x) if pd.notna(x) else False)

        group_remove_mask = data["titre_log_diff_group_removed_by_violate_rate"].eq(True)
        target_remove_mask = data["target_row_removed_by_low_self_reference_group"].eq(True)
        before = len(data)
        metrics[f"{source_prefix}_rows_removed_by_titre_log_diff_violate_rate"] = int(group_remove_mask.sum())
        metrics[f"{source_prefix}_rows_removed_by_low_self_reference_target_filter"] = int(
            (target_remove_mask & ~group_remove_mask).sum()
        )
        data = data[~(group_remove_mask | target_remove_mask)].copy()
        metrics[f"{source_prefix}_after_titre_log_diff_violate_rate_filter_rows"] = int(len(data))
        metrics[f"{source_prefix}_after_low_self_reference_target_filter_rows"] = int(len(data))
        data = data.drop(columns=["titre_log_diff_is_violation"], errors="ignore")

    # Match the previous notebook logic: remove rows whose titre_log_diff is < -1
    # before converting negative values to zero for score calculation.
    before = len(data)
    data = data[data["titre_log_diff"] >= -1].copy()
    metrics[f"{source_prefix}_dropped_rows_titre_log_diff_lt_minus1"] = int(before - len(data))
    metrics[f"{source_prefix}_after_titre_log_diff_ge_minus1_filter_rows"] = int(len(data))

    data["titre_log_diff_nonnegative"] = data["titre_log_diff"].clip(lower=0)
    data["score"] = data["titre_log_diff_nonnegative"].clip(upper=fixed_score_scale) / fixed_score_scale

    metrics[f"{source_prefix}_fixed_score_scale"] = float(fixed_score_scale)
    metrics[f"{source_prefix}_max_titre_log_diff_nonnegative_final"] = (
        float(data["titre_log_diff_nonnegative"].max()) if not data.empty else None
    )
    metrics[f"{source_prefix}_n_rows_score_clipped_at_fixed_scale"] = (
        int((data["titre_log_diff_nonnegative"] > fixed_score_scale).sum()) if not data.empty else 0
    )
    metrics[f"{source_prefix}_final_scored_rows"] = int(len(data))

    requested_columns = [
        "data_source",
        "season",
        "date",
        "virus",
        "reference",
        "virus_passage",
        "reference_passage",
        "virus_collection_date",
        "score",
        "censor",
        "virus_seq",
        "reference_seq",
    ]

    qc_columns = [
        "source_file",
        "sheet_name",
        "date_is_season_fallback",
        "date_raw",
        "date_is_invalid_preserved",
        "virus_passage_raw",
        "reference_passage_raw",
        "virus_passage_type",
        "reference_passage_type",
        "HI_titre",
        "HI_raw",
        "titre_log",
        "self_mean_titre_log",
        "titre_log_diff",
        "titre_log_diff_group_n",
        "titre_log_diff_violate_n",
        "titre_log_diff_violate_rate",
        "titre_log_diff_viorate_rate",
        "titre_log_diff_group_removed_by_violate_rate",
        "reference_group_self_mean_titre_log",
        "date_median_self_mean_titre_log",
        "low_self_reference_group",
        "target_row_removed_by_low_self_reference_group",
        "titre_log_diff_nonnegative",
        "virus_seq_X_count",
        "reference_seq_X_count",
        "reference_from_header",
        "reference_from_row",
        "reference_match_score",
        "virus_collection_date_source",
        "virus_collection_date_is_inferred_from_name",
        "virus_fasta_collection_date",
        "reference_fasta_collection_date",
        "virus_strain_passage",
        "reference_strain_passage",
        "virus_fasta_matched",
        "reference_fasta_matched",
        "virus_raw",
        "reference_raw",
        "virus_clean_for_match",
        "reference_clean_for_match",
        "antiserum",
        "RBC",
        "pair",
        "pair_strict",
        "external_row",
    ]

    ordered = [c for c in requested_columns + qc_columns if c in data.columns]
    remaining = [c for c in data.columns if c not in ordered]
    data = data[ordered + remaining].reset_index(drop=True)

    return data, build_log(metrics)


# -----------------------------
# Combining
# -----------------------------

def resolve_paths(inputs: Iterable[str] | None) -> list[Path]:
    if not inputs:
        return []

    paths = []
    for item in inputs:
        item = str(item)
        expanded = Path(item).expanduser()

        if any(ch in item for ch in "*?[]"):
            matches = [Path(x) for x in glob.glob(item)]
        elif expanded.is_file():
            matches = [expanded]
        elif expanded.is_dir():
            matches = list(expanded.glob("*.csv"))
        else:
            matches = [Path(x) for x in glob.glob(item)]

        paths.extend([p for p in matches if p.suffix.lower() == ".csv"])

    return sorted({str(p): p for p in paths}.values(), key=lambda x: str(x))


def read_who_score_csvs(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        if "data_source" not in df.columns:
            df.insert(0, "data_source", "who_hi_extractor")
        if "source_file" not in df.columns:
            df["source_file"] = path.name

        # Recalculate WHO-extractor CSV seasons from valid assay dates using the
        # updated NH naming convention. This prevents old extractor outputs from
        # carrying old-style NH labels into the combined table. Missing/invalid
        # dates keep their existing season value if one is present.
        if "date" in df.columns:
            if "season" not in df.columns:
                df["season"] = pd.NA
            derived = df["date"].apply(derive_season_from_date)
            mask = derived.notna()
            df.loc[mask, "season"] = derived[mask]

        frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True, sort=False)


def combine_outputs(
    external_scored: pd.DataFrame,
    who_scored: pd.DataFrame,
    drop_duplicates: bool = True,
) -> pd.DataFrame:
    frames = []
    if who_scored is not None and not who_scored.empty:
        frames.append(who_scored)
    if external_scored is not None and not external_scored.empty:
        frames.append(external_scored)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True, sort=False)

    if drop_duplicates:
        subset = [
            c for c in [
                "date",
                "virus",
                "reference",
                "virus_passage",
                "reference_passage",
                "HI_titre",
                "score",
                "virus_seq",
                "reference_seq",
            ] if c in combined.columns
        ]
        combined = combined.drop_duplicates(subset=subset).reset_index(drop=True)

    requested_columns = [
        "data_source",
        "season",
        "date",
        "virus",
        "reference",
        "virus_passage",
        "reference_passage",
        "virus_collection_date",
        "score",
        "censor",
        "virus_seq",
        "reference_seq",
    ]
    ordered = [c for c in requested_columns if c in combined.columns]
    remaining = [c for c in combined.columns if c not in ordered]
    return combined[ordered + remaining]


def add_name_normalization_checks(metrics: dict):
    """Add explicit checks for year truncation behavior."""
    examples = {
        "name_check_A_Kamata_85_1987": "A/Kamata/85/1987",
        "name_check_A_Tokyo_1_51": "A/Tokyo/1/51",
        "name_check_A_Memphis_102_72": "A/Memphis/102/72",
        "name_check_A_Wisconsin_67_2005_ISOLATE2": "A_WISCONSIN_67_2005#ISOLATE2",
        "name_check_A_Tokyo_1_51_extra_note": "A/Tokyo/1/51 extra_note",
        "name_check_A_Wisconsin_67_2005_direct_suffix": "A/WISCONSIN/67/2005ISOLATE2",
        "name_check_IVR_238_parenthetical_suffix": "IVR-238 (A/Victoria/4897/2022)#note",
        "name_check_A_Trieste_25c_2007": "A/Trieste/25c/2007",
        "name_check_A_TRIESTE_25C_2007": "A/TRIESTE/25C/2007",
    }
    for key, value in examples.items():
        metrics[key] = output_format_strain_name(value)
        metrics[f"{key}_underscore_match_key"] = canonical_strain_name(value)


# -----------------------------
# CLI
# -----------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Score/link an external H3N2 HI table with FASTA sequences and combine "
            "it with scored outputs from who_hi_extractor_fasta_auto.py."
        )
    )

    parser.add_argument(
        "--external-hi",
        required=True,
        help="External HI-titre CSV table with columns virus, reference, virus_passage, reference_passage, date, titre.",
    )
    parser.add_argument(
        "--who-score-csv",
        nargs="*",
        default=[],
        help="Existing scored CSV output(s) from who_hi_extractor_fasta_auto.py. Glob patterns are allowed.",
    )
    parser.add_argument(
        "--fasta",
        required=True,
        help="FASTA file used to link sequences. Supports both H1N1-style and H3N2-style headers.",
    )
    parser.add_argument(
        "--out",
        default="combined_HI_score.csv",
        help="Combined output CSV path.",
    )
    parser.add_argument(
        "--external-out",
        default="external_H3N2_HI_scored.csv",
        help="Output CSV path for scored external HI table only.",
    )
    parser.add_argument(
        "--missing-fasta-out",
        default="external_H3N2_missing_fasta_matches.csv",
        help=(
            "Output CSV path for the unique virus/reference strain names from the "
            "external HI table that could not be matched to the FASTA file."
        ),
    )
    parser.add_argument(
        "--log",
        default="combined_HI_score_log.csv",
        help="Output CSV path for QC metrics.",
    )
    parser.add_argument(
        "--score-scale",
        type=float,
        default=8.0,
        help="Fixed scale used to convert nonnegative titre-log difference to score. Default: 8.0",
    )
    parser.add_argument(
        "--titre-log-diff-violate-cutoff",
        type=float,
        default=-1.0,
        help=(
            "Cutoff used for group-level QC. Rows with titre_log_diff <= this value "
            "are counted as violations. Default: -1.0"
        ),
    )
    parser.add_argument(
        "--titre-log-diff-violate-rate-threshold",
        type=float,
        default=0.5,
        help=(
            "Remove all rows from a date/reference/reference_passage group when the "
            "fraction of rows with titre_log_diff <= --titre-log-diff-violate-cutoff "
            "is greater than this threshold. Default: 0.5"
        ),
    )
    parser.add_argument(
        "--remove-target-rows-linked-to-low-self-reference-groups",
        dest="remove_target_rows_linked_to_low_self_reference_groups",
        action="store_true",
        default=True,
        help=(
            "Also remove rows where a low-self bad reference strain appears as the "
            "target virus in the same date/passage context. This is enabled by default."
        ),
    )
    parser.add_argument(
        "--no-remove-target-rows-linked-to-low-self-reference-groups",
        dest="remove_target_rows_linked_to_low_self_reference_groups",
        action="store_false",
        help="Disable target-side removal linked to low-self bad reference groups.",
    )
    parser.add_argument(
        "--low-self-titre-log-margin",
        type=float,
        default=2.0,
        help=(
            "A bad reference group is considered low-self when its self_mean_titre_log "
            "is at least this many log2 units below the same-date median self_mean_titre_log. "
            "Default: 2.0"
        ),
    )
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Do not drop duplicate rows after combining outputs.",
    )

    args = parser.parse_args(argv)

    external_path = Path(args.external_hi).expanduser()
    fasta_path = Path(args.fasta).expanduser()

    if not external_path.is_file():
        print(f"External HI table was not found: {external_path}", file=sys.stderr)
        return 1

    if not fasta_path.is_file():
        print(f"FASTA file was not found: {fasta_path}", file=sys.stderr)
        return 1

    who_score_paths = resolve_paths(args.who_score_csv)

    fasta_index = read_fasta_index(fasta_path)
    external_raw = read_external_hi_table(external_path)
    external_long, normalize_metrics = normalize_external_hi_table(
        external_raw,
        fasta_index=fasta_index,
        source_name=external_path.name,
    )
    missing_fasta_matches = build_missing_fasta_match_table(external_long)

    external_scored, score_log = build_scored_filtered_dataset(
        external_long,
        fixed_score_scale=args.score_scale,
        source_prefix="external",
        titre_log_diff_violate_cutoff=args.titre_log_diff_violate_cutoff,
        titre_log_diff_violate_rate_threshold=args.titre_log_diff_violate_rate_threshold,
        remove_target_rows_linked_to_low_self_reference_groups=(
            args.remove_target_rows_linked_to_low_self_reference_groups
        ),
        low_self_titre_log_margin=args.low_self_titre_log_margin,
    )

    who_scored = read_who_score_csvs(who_score_paths)

    combined = combine_outputs(
        external_scored,
        who_scored,
        drop_duplicates=not args.keep_duplicates,
    )

    metrics = {
        "fasta_path": str(fasta_path),
        "fasta_records": fasta_index["n_records"],
        "fasta_records_with_collection_date": fasta_index.get("n_records_with_collection_date", 0),
        "fasta_unique_canonical_names": len(fasta_index["seq_by_canon"]),
        "fasta_duplicate_canonical_names_ignored": fasta_index["n_duplicate_canonical_names"],
        "external_hi_path": str(external_path),
        "who_score_csv_count": len(who_score_paths),
        "who_score_csv_rows": int(len(who_scored)),
        "external_scored_rows": int(len(external_scored)),
        "missing_fasta_match_unique_names": int(len(missing_fasta_matches)),
        "missing_fasta_match_unique_virus_names": int((missing_fasta_matches["match_side"] == "virus").sum()) if not missing_fasta_matches.empty else 0,
        "missing_fasta_match_unique_reference_names": int((missing_fasta_matches["match_side"] == "reference").sum()) if not missing_fasta_matches.empty else 0,
        "combined_rows": int(len(combined)),
    }
    metrics.update(normalize_metrics)
    add_name_normalization_checks(metrics)

    combined_log = pd.concat(
        [build_log(metrics), score_log],
        ignore_index=True,
    )

    external_scored.to_csv(args.external_out, index=False)
    missing_fasta_matches.to_csv(args.missing_fasta_out, index=False)
    combined.to_csv(args.out, index=False)
    combined_log.to_csv(args.log, index=False)

    print(f"External scored rows: {len(external_scored):,}")
    print(f"WHO scored rows: {len(who_scored):,}")
    print(f"Combined rows: {len(combined):,}")
    print(f"External scored data written to: {args.external_out}")
    print(f"Missing FASTA match strain list written to: {args.missing_fasta_out}")
    print(f"Combined data written to: {args.out}")
    print(f"Log written to: {args.log}")
    print("Name normalization checks:")
    print(f"  A/Kamata/85/1987 -> {output_format_strain_name('A/Kamata/85/1987')}")
    print(f"  A/Tokyo/1/51     -> {output_format_strain_name('A/Tokyo/1/51')}")
    print(f"  A/Memphis/102/72 -> {output_format_strain_name('A/Memphis/102/72')}")
    print(f"  A_WISCONSIN_67_2005#ISOLATE2 -> {output_format_strain_name('A_WISCONSIN_67_2005#ISOLATE2')}")
    print(f"  A/Trieste/25c/2007 -> {output_format_strain_name('A/Trieste/25c/2007')}")
    print("Underscore matching-key checks:")
    print(f"  A/Trieste/25c/2007 -> {canonical_strain_name('A/Trieste/25c/2007')}")
    print(f"  A_TRIESTE_25C_2007 -> {canonical_strain_name('A_TRIESTE_25C_2007')}")
    print(f"  A/Tokyo/1/51 extra_note -> {output_format_strain_name('A/Tokyo/1/51 extra_note')}")
    print(f"  A/Trieste/25c/2007 -> {output_format_strain_name('A/Trieste/25c/2007')}")
    print(f"  A_WISCONSIN_67_2005ISOLATE2 -> {output_format_strain_name('A_WISCONSIN_67_2005ISOLATE2')}")
    print(f"  IVR-238 (A/Victoria/4897/2022)#note -> {output_format_strain_name('IVR-238 (A/Victoria/4897/2022)#note')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
