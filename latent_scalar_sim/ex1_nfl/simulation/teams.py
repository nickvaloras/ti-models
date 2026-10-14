"""Team definitions and colors.

The order of TEAMS is the *true* hierarchy (index 0 = strongest), mirroring the
ordered list A > B > C > ... of a transitive-inference task.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Team:
    abbr: str
    name: str
    primary: str    # main fill color
    secondary: str  # edge / marker accent


TEAMS: list[Team] = [
    Team("NE",  "New England Patriots",  "#002244", "#C60C30"),
    Team("KC",  "Kansas City Chiefs",    "#E31837", "#FFB81C"),
    Team("PHI", "Philadelphia Eagles",   "#004C54", "#A5ACAF"),
    Team("GB",  "Green Bay Packers",     "#203731", "#FFB612"),
    Team("SEA", "Seattle Seahawks",      "#69BE28", "#002244"),
    Team("PIT", "Pittsburgh Steelers",   "#FFB612", "#101820"),
    Team("DAL", "Dallas Cowboys",        "#869397", "#003594"),
    Team("DEN", "Denver Broncos",        "#FB4F14", "#002244"),
    Team("CHI", "Chicago Bears",         "#0B162A", "#C83803"),
    Team("MIN", "Minnesota Vikings",     "#4F2683", "#FFC62F"),
    Team("NO",  "New Orleans Saints",    "#D3BC8D", "#101820"),
    Team("NYG", "New York Giants",       "#0B2265", "#A71930"),
    Team("HOU", "Houston Texans",        "#03202F", "#A71930"),
    Team("ARI", "Arizona Cardinals",     "#97233F", "#FFB612"),
    Team("NYJ", "New York Jets",         "#125740", "#000000"),
    Team("CLE", "Cleveland Browns",      "#311D00", "#FF3C00"),
]

N_TEAMS = len(TEAMS)
ABBRS = [t.abbr for t in TEAMS]
INDEX = {t.abbr: i for i, t in enumerate(TEAMS)}


def text_color_on(hex_color: str) -> str:
    """Black or white text, whichever reads better on the given fill."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))

    def lin(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    lum = 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)
    return "#111111" if lum > 0.35 else "#FFFFFF"
