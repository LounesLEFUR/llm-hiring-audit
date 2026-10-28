"""Check every first name against the INSEE first-names file.

Usage:
    uv run python scripts/check_first_names.py prenoms-2025-nat.csv

The national CSV is published by INSEE ("Fichier des prénoms", Open Licence):
https://www.insee.fr/fr/statistiques/8595130

Prints, for each group, the number of children given each name in France
between 1985 and 2000 (candidates aged about 25-40 at the time of the audit),
and exits with an error if a name is rare or unknown.
"""

import sys
import unicodedata

import pandas as pd

from llm_hiring_audit.generator import _load_json

FIRST_YEAR, LAST_YEAR = 1985, 2000
MIN_BIRTHS = 100
INSEE_SEX = {"male": 1, "female": 2}


def fold(name: str) -> str:
    """Upper case without accents: INSEE spells some names without them."""
    decomposed = unicodedata.normalize("NFKD", name)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).upper()


def main(csv_path: str) -> int:
    df = pd.read_csv(csv_path, sep=";", dtype={"periode": str, "prenom": str})
    df = df[df["periode"].str.fullmatch(r"\d{4}")]
    df = df[df["periode"].astype(int).between(FIRST_YEAR, LAST_YEAR)]
    births = df.assign(prenom=df["prenom"].map(fold)).groupby(["sexe", "prenom"])["valeur"].sum()

    rare = []
    for origin, by_gender in _load_json("names.json")["first_names"].items():
        for gender, names in by_gender.items():
            counts = [(name, int(births.get((INSEE_SEX[gender], fold(name)), 0))) for name in names]
            print(f"{origin} {gender}: " + ", ".join(f"{n} {c:,}" for n, c in counts))
            rare += [n for n, c in counts if c < MIN_BIRTHS]

    if rare:
        print(f"\nFewer than {MIN_BIRTHS} births {FIRST_YEAR}-{LAST_YEAR}: {', '.join(rare)}")
        return 1
    print(f"\nAll first names were given at least {MIN_BIRTHS} times in {FIRST_YEAR}-{LAST_YEAR}.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
