#!/usr/bin/env python3
"""Build grouped Polish ratings CSV for hybrid SSR tuning.

Source dataset:
- allegro/klej-allegro-reviews (text + rating)

Output format matches tune_ssr_hierarchical.py expectations:
- group,text,label,country,industry,language,source,split,notes
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path

import numpy as np
from datasets import load_dataset


# Coarse taxonomy (legacy).
POLISH_GROUP_RULES_COARSE: list[tuple[str, tuple[str, ...]]] = [
    (
        "PL_Electronics",
        (
            "telefon",
            "smartfon",
            "laptop",
            "komputer",
            "monitor",
            "kabel",
            "słuchawk",
            "sluchawk",
            "głośnik",
            "glosnik",
            "mysz",
            "klawiatur",
            "ładowark",
            "ladowark",
            "usb",
            "bluetooth",
            "tv",
            "telewiz",
            "router",
            "drukark",
            "powerbank",
        ),
    ),
    (
        "PL_Beauty_and_Personal_Care",
        (
            "szampon",
            "odżywk",
            "odzywk",
            "krem",
            "serum",
            "paznok",
            "makija",
            "kosmet",
            "perfum",
            "zapach",
            "tusz",
            "żel pod prysznic",
            "zel pod prysznic",
            "higien",
            "depila",
            "goleni",
        ),
    ),
    (
        "PL_Health_and_Personal_Care",
        (
            "witamin",
            "suplement",
            "tablet",
            "kapsuł",
            "kapsul",
            "ból",
            "bol",
            "zdrow",
            "leczen",
            "termometr",
            "ciśnieniomierz",
            "cisnieniomierz",
            "apte",
        ),
    ),
    (
        "PL_Home_and_Kitchen",
        (
            "kuch",
            "garnek",
            "pateln",
            "mikser",
            "blender",
            "czajnik",
            "pościel",
            "posciel",
            "podusz",
            "kołdr",
            "koldr",
            "ręcznik",
            "recznik",
            "mebel",
            "dywan",
            "lamp",
            "odkurzacz",
            "pralka",
            "lodówk",
            "lodowk",
        ),
    ),
    (
        "PL_Clothing_Shoes_and_Jewelry",
        (
            "koszul",
            "spodni",
            "sukienk",
            "but",
            "obuw",
            "kurtk",
            "bluzy",
            "biżut",
            "bizut",
            "pierścion",
            "pierscion",
            "zegarek",
            "torb",
            "plecak",
        ),
    ),
    (
        "PL_Automotive",
        (
            "samoch",
            "auto",
            "opon",
            "felg",
            "olej silnik",
            "hamul",
            "akumulator",
            "nawigac",
            "uchwyt samochod",
        ),
    ),
    (
        "PL_Sports_and_Outdoors",
        (
            "rower",
            "siłown",
            "silown",
            "trening",
            "fitness",
            "piłk",
            "pilk",
            "bieg",
            "namiot",
            "turyst",
            "kemping",
            "hantel",
        ),
    ),
    (
        "PL_Toys_and_Games",
        (
            "zabawk",
            "klock",
            "gra plansz",
            "puzzle",
            "dla dziec",
            "lalka",
            "samochodzik",
            "maskot",
        ),
    ),
    (
        "PL_Pet_Supplies",
        (
            "pies",
            "kot",
            "karma",
            "kuweta",
            "smycz",
            "obroż",
            "obroz",
            "zwierzę",
            "zwierze",
        ),
    ),
    (
        "PL_Office_Products",
        (
            "biur",
            "notes",
            "zeszyt",
            "długopis",
            "dlugopis",
            "papier",
            "segregator",
            "toner",
            "tusz do drukarki",
        ),
    ),
    (
        "PL_Tools_and_Home_Improvement",
        (
            "wiertark",
            "wkrętark",
            "wkretark",
            "śrub",
            "srub",
            "młotek",
            "mlotek",
            "narzęd",
            "narzed",
            "remont",
            "farb",
        ),
    ),
]


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _assign_group_coarse(text: str) -> tuple[str, str]:
    t = _norm(text)
    for group, keywords in POLISH_GROUP_RULES_COARSE:
        for kw in keywords:
            if kw in t:
                # group format: PL_<industry>
                return group, group.replace("PL_", "", 1)
    return "PL_Mixed", "Mixed"


# More precise taxonomy, aligned with 20-industry style.
POLISH_GROUP_RULES_PRECISE: list[tuple[str, tuple[str, ...]]] = [
    ("PL_Electronics", ("telefon", "smartfon", "laptop", "komputer", "monitor", "kabel", "usb", "hdmi", "ładowark", "ladowark", "powerbank", "drukark", "router", "kamera", "aparat", "słuchawk", "sluchawk", "głośnik", "glosnik", "tv", "telewiz")),
    ("PL_Books", ("książk", "ksiazk", "powieś", "powies", "podręcznik", "podrecznik", "ebook", "audiobook", "autor", "czyta")),
    ("PL_Clothing_Shoes_and_Jewelry", ("koszul", "spodni", "spodnie", "sukienk", "but", "obuw", "kurtk", "bluzy", "sweter", "biżut", "bizut", "pierścion", "pierscion", "bransolet", "naszyjnik", "zegarek", "torebk", "plecak")),
    ("PL_Home_and_Kitchen", ("kuch", "garnek", "pateln", "sztućce", "sztucce", "czajnik", "mikser", "blender", "odkurzacz", "zmywark", "pralka", "lodówk", "lodowk", "pościel", "posciel", "podusz", "kołdr", "koldr", "dywan", "lamp", "mebel", "materac", "ręcznik", "recznik")),
    ("PL_Automotive", ("samoch", "auto", "opon", "felg", "akumulator", "hamul", "olej silnik", "wycieraczk", "uchwyt samochod", "nawigac", "motoryzac")),
    ("PL_Sports_and_Outdoors", ("rower", "siłown", "silown", "fitness", "trening", "bieg", "biegan", "piłk", "pilk", "hantel", "mata do ćwiczeń", "mata do cwiczen", "namiot", "śpiwór", "spiwor", "turyst", "kemping")),
    ("PL_Health_and_Personal_Care", ("witamin", "suplement", "tablet", "kapsuł", "kapsul", "zdrow", "ciśnieniomierz", "cisnieniomierz", "termometr", "ból", "bol", "leczen", "rehabilit", "apte")),
    ("PL_Beauty_and_Personal_Care", ("szampon", "odżywk", "odzywk", "krem", "serum", "kosmet", "makija", "perfum", "zapach", "tusz", "paznok", "depila", "goleni", "żel pod prysznic", "zel pod prysznic")),
    ("PL_Toys_and_Games", ("zabawk", "klock", "gra plansz", "planszówk", "planszowk", "puzzle", "dla dziec", "lalka", "maskot", "samochodzik", "lego")),
    ("PL_Video_Games", ("playstation", "xbox", "nintendo", "konsol", "gra komputer", "gra ps", "steam", "pad", "kontroler", "gaming")),
    ("PL_Office_Products", ("biur", "notes", "zeszyt", "długopis", "dlugopis", "papier", "segregator", "toner", "tusz do drukarki", "spinacz", "kalkulator", "marker", "zakreślacz", "zakreslacz")),
    ("PL_Pet_Supplies", ("pies", "kot", "karma", "kuweta", "smycz", "obroż", "obroz", "zwierzę", "zwierze", "drapak", "legowisk")),
    ("PL_Grocery_and_Gourmet_Food", ("kawa", "herbat", "czekolad", "makaron", "oliw", "przypraw", "spożywc", "spozywc", "żywno", "zywno", "smak", "napój", "napoj", "baton", "ciastk")),
    ("PL_Appliances", ("mikrofal", "ekspres do kawy", "toster", "opiekacz", "robot kuchenny", "suszark", "żelazk", "zelazk", "frytkownic", "airfryer")),
    ("PL_Tools_and_Home_Improvement", ("wiertark", "wkrętark", "wkretark", "śrub", "srub", "młotek", "mlotek", "narzęd", "narzed", "remont", "farb", "taśma miernicza", "tasma miernicza", "poziomic", "kombinerk")),
    ("PL_Cell_Phones_and_Accessories", ("etui", "szkło hartowane", "szklo hartowane", "case", "ładowarka do telefonu", "ladowarka do telefonu", "uchwyt do telefonu", "kabel lightning", "iphone", "samsung galaxy")),
    ("PL_Musical_Instruments", ("gitara", "ukulele", "pianino", "keyboard", "klawisz", "mikrofon", "wzmacniacz", "nuty", "instrument", "perkus")),
    ("PL_Software", ("oprogramowanie", "licencja", "antywirus", "program", "aplikacja", "subskrypcja", "windows", "office 365", "edytor", "plugin")),
    ("PL_Patio_Lawn_and_Garden", ("ogród", "ogrod", "trawnik", "kosiark", "grill", "donicz", "nawadnian", "altana", "taras", "sekator", "gleba")),
    ("PL_Industrial_and_Scientific", ("laborator", "pomiar", "miernik", "oscyloskop", "techniczny", "przemysł", "przemysl", "inżynier", "inzynier", "kalibrac", "czujnik")),
]


def _assign_group_precise(text: str, min_score: int = 1) -> tuple[str, str]:
    t = _norm(text)
    scores: dict[str, int] = {}
    for group, keywords in POLISH_GROUP_RULES_PRECISE:
        score = 0
        for kw in keywords:
            if kw in t:
                score += 1
        if score > 0:
            scores[group] = score
    if not scores:
        return "PL_Mixed", "Mixed"

    # Select highest score; deterministic tie-break by group name.
    best_group = sorted(scores.items(), key=lambda x: (-x[1], x[0]))[0][0]
    if scores[best_group] < min_score:
        return "PL_Mixed", "Mixed"
    return best_group, best_group.replace("PL_", "", 1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build grouped CSV for Polish ratings.")
    p.add_argument(
        "--dataset",
        default="allegro/klej-allegro-reviews",
        help="HF dataset id (default: allegro/klej-allegro-reviews).",
    )
    p.add_argument(
        "--split",
        default="train+validation+test",
        help="HF split expression.",
    )
    p.add_argument(
        "--sample-per-group",
        type=int,
        default=1200,
        help="Max rows per group in output (0 = keep all).",
    )
    p.add_argument(
        "--min-group-n",
        type=int,
        default=300,
        help="Drop groups smaller than this threshold (except PL_Mixed).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--taxonomy",
        choices=["coarse", "precise"],
        default="precise",
        help="Grouping taxonomy complexity.",
    )
    p.add_argument(
        "--min-keyword-score",
        type=int,
        default=1,
        help="For precise taxonomy: minimum matched keyword count required.",
    )
    p.add_argument(
        "--out",
        default="/Users/pawel/Market_wizard/reports/pl_grouped_ratings_2026-02-07.csv",
        help="Output CSV path.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    ds = load_dataset(args.dataset, split=args.split)
    rows_by_group: dict[str, list[tuple[str, int, str, str]]] = {}

    # Support both rating/label column names.
    label_col = "rating" if "rating" in ds.column_names else "label"
    if label_col not in ds.column_names or "text" not in ds.column_names:
        raise RuntimeError(
            f"Dataset must contain text + rating/label columns. Found: {ds.column_names}"
        )

    for item in ds:
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        raw = item.get(label_col)
        try:
            rating = int(round(float(raw)))
        except Exception:
            continue
        if rating < 1 or rating > 5:
            continue
        if args.taxonomy == "coarse":
            group, industry = _assign_group_coarse(text)
        else:
            group, industry = _assign_group_precise(text, min_score=args.min_keyword_score)
        rows_by_group.setdefault(group, []).append((text, rating, industry, "allegro"))

    kept_groups: dict[str, list[tuple[str, int, str, str]]] = {}
    for group, rows in rows_by_group.items():
        if group != "PL_Mixed" and len(rows) < args.min_group_n:
            continue
        rng.shuffle(rows)
        if args.sample_per_group > 0:
            rows = rows[: args.sample_per_group]
        kept_groups[group] = rows

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["group", "text", "label", "country", "industry", "language", "source", "split", "notes"]
        )
        for group in sorted(kept_groups):
            rows = kept_groups[group]
            for text, rating, industry, source in rows:
                writer.writerow([group, text, rating, "PL", industry, "pl", source, "all", f"taxonomy={args.taxonomy}"])

    counts = {g: len(v) for g, v in kept_groups.items()}
    total = sum(counts.values())
    print(f"Saved: {out_path}")
    print(f"Groups kept: {len(counts)} | rows: {total}")
    for g, n in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        print(f"- {g}: {n}")

    # Print label distribution for sanity.
    labels = Counter()
    for rows in kept_groups.values():
        for _, rating, _, _ in rows:
            labels[rating] += 1
    print("Label distribution:")
    for i in range(1, 6):
        print(f"- {i}: {labels.get(i, 0)}")


if __name__ == "__main__":
    main()
