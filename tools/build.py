#!/usr/bin/env python3
"""Build the profile images: the caffeine hero and the Karte menu card.

    python tools/build.py

Writes assets/hero-{dark,light}.svg and the Die Karte strips assets/karte-*-{dark,light}.svg
(one header strip plus one linkable strip per project).
All text is outlined to paths (no fonts are embedded or loaded by the viewer).
Needs fontTools; the fonts live in tools/fonts/ (SIL OFL, see the OFL.txt files).
The molecule is computed from ring geometry and checked against its own atom and
bond data before anything is written; a failed check stops the build.
"""

from __future__ import annotations

import math
import sys
import tomllib
from collections import Counter
from functools import lru_cache
from pathlib import Path

from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.ttLib import TTFont
from fontTools.varLib import instancer

ROOT = Path(__file__).resolve().parent.parent
FONTS = ROOT / "tools" / "fonts"
ASSETS = ROOT / "assets"
DATA = ROOT / "data" / "projects.toml"

W = 830  # GitHub's desktop content column

PALETTES = {
    "dark": {
        "bg": "#17110D",  # espresso
        "ink": "#EFE4D2",  # cream
        "accent": "#D0596A",  # velvet
        "brass": "#CDAA6E",
        "muted": "#9C8B78",
    },
    "light": {
        "bg": "#F5EFE4",  # menu paper
        "ink": "#2B1D14",  # roast
        "accent": "#8C2333",  # burgundy
        "brass": "#9A7740",
        "muted": "#7A6A58",
    },
}


def r(v: float) -> str:
    """Round for stable output (reruns produce no diff)."""
    s = f"{v:.1f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


# --------------------------------------------------------------------------- text


@lru_cache(maxsize=None)
def face(file: str, **axes: float):
    font = TTFont(FONTS / file)
    if "fvar" in font:
        font = instancer.instantiateVariableFont(font, axes)
    return font


class Text:
    """One font at one size: measure and outline strings."""

    def __init__(self, file: str, size: float, tracking: float = 0.0, **axes: float):
        self.font = face(file, **axes)
        self.key = (file, tuple(sorted(axes.items())))
        self.size = size
        self.tracking = tracking  # extra advance, in em
        self.scale = size / self.font["head"].unitsPerEm
        self.cmap = self.font.getBestCmap()
        self.glyphs = self.font.getGlyphSet()
        self.hmtx = self.font["hmtx"]

    def _names(self, s: str) -> list[str]:
        missing = [c for c in s if ord(c) not in self.cmap]
        assert not missing, f"glyphs missing for {missing!r} in {s!r}"
        return [self.cmap[ord(c)] for c in s]

    def width(self, s: str) -> float:
        names = self._names(s)
        adv = sum(self.hmtx[n][0] for n in names) * self.scale
        return adv + self.tracking * self.size * max(len(names) - 1, 0)

    def uses(self, s: str, x: float, y: float, anchor: str = "start") -> str:
        """s as <use> references to glyphs outlined once into DEFS (keeps files small)."""
        if anchor == "middle":
            x -= self.width(s) / 2
        elif anchor == "end":
            x -= self.width(s)
        out = []
        sc = f"{self.scale:.6g}"
        for name in self._names(s):
            key = (self.key, name)
            if key not in DEFS:
                pen = SVGPathPen(self.glyphs, ntos=lambda v: str(round(v)))
                self.glyphs[name].draw(pen)
                DEFS[key] = (f"g{len(DEFS)}", pen.getCommands())
            gid, d = DEFS[key]
            if d:  # spaces have no outline
                out.append(f'<use href="#{gid}" transform="translate({r(x)} {r(y)}) scale({sc} -{sc})"/>')
            x += self.hmtx[name][0] * self.scale + self.tracking * self.size
        return "".join(out)


DEFS: dict = {}  # glyph outlines of the document being built

FRAUNCES = "fraunces/Fraunces.ttf"
FRAUNCES_I = "fraunces/Fraunces-Italic.ttf"
PLEX = "ibm-plex-mono/IBMPlexMono-Medium.ttf"


def fraunces(size, wght=480, opsz=None, soft=50, wonk=0, italic=False, tracking=0.0):
    return Text(
        FRAUNCES_I if italic else FRAUNCES,
        size,
        tracking,
        wght=wght,
        opsz=min(max(opsz or size, 9), 144),
        SOFT=soft,
        WONK=wonk,
    )


def plex(size, tracking=0.0):
    return Text(PLEX, size, tracking)


# ----------------------------------------------------------------------- molecule

L = 60.0  # bond length, viewBox units


def build_caffeine(cx: float, cy: float):
    """Caffeine (1,3,7-trimethylxanthine) on a regular hexagon + pentagon.

    Pointy-top hexagon, C5-C4 the vertical edge on the right; the imidazole
    pentagon is fused on that edge. Purine numbering.
    """
    def polar(ox, oy, ang, d):
        a = math.radians(ang)
        return (ox + d * math.cos(a), oy + d * math.sin(a))  # y points down

    atoms: dict[str, dict] = {}

    def add(name, el, pos):
        atoms[name] = {"el": el, "pos": pos}

    hexa = {"C5": -30, "C6": -90, "N1": -150, "C2": 150, "N3": 90, "C4": 30}
    for name, ang in hexa.items():
        add(name, name[0], polar(cx, cy, ang, L))

    # pentagon on edge C5-C4
    apothem = L / (2 * math.tan(math.radians(36)))
    circum = L / (2 * math.sin(math.radians(36)))
    edge_x = atoms["C5"]["pos"][0]
    px, py = edge_x + apothem, cy
    a_c5 = math.degrees(math.atan2(atoms["C5"]["pos"][1] - py, atoms["C5"]["pos"][0] - px))
    # walk C5 -> N7 -> C8 -> N9 -> C4, i.e. clockwise on screen (+72 deg, y down)
    for i, name in enumerate(["N7", "C8", "N9"], start=1):
        add(name, name[0], polar(px, py, a_c5 + 72 * i, circum))

    def outward(atom, centre, d=L):
        ax, ay = atoms[atom]["pos"]
        ang = math.degrees(math.atan2(ay - centre[1], ax - centre[0]))
        return polar(ax, ay, ang, d)

    add("O6", "O", outward("C6", (cx, cy)))
    add("O2", "O", outward("C2", (cx, cy)))
    add("Me1", "C", outward("N1", (cx, cy)))
    add("Me3", "C", outward("N3", (cx, cy)))
    add("Me7", "C", outward("N7", (px, py)))

    bonds = [
        ("N1", "C2", 1), ("C2", "N3", 1), ("N3", "C4", 1), ("C4", "C5", 2),
        ("C5", "C6", 1), ("C6", "N1", 1),
        ("C5", "N7", 1), ("N7", "C8", 1), ("C8", "N9", 2), ("N9", "C4", 1),
        ("C2", "O2", 2), ("C6", "O6", 2),
        ("N1", "Me1", 1), ("N3", "Me3", 1), ("N7", "Me7", 1),
    ]
    rings = {
        "hexagon": (["N1", "C2", "N3", "C4", "C5", "C6"], (cx, cy)),
        "pentagon": (["C4", "C5", "N7", "C8", "N9"], (px, py)),
    }
    return atoms, bonds, rings


def check_caffeine(atoms, bonds, rings) -> list[str]:
    log = []
    valence = {"C": 4, "N": 3, "O": 2}
    order = Counter()
    nbrs: dict[str, set] = {a: set() for a in atoms}
    for a, b, o in bonds:
        order[a] += o
        order[b] += o
        nbrs[a].add(b)
        nbrs[b].add(a)

    def bonded(a, b):
        return b in nbrs[a]

    hydrogens = {}
    for name, atom in atoms.items():
        used = order[name]
        v = valence[atom["el"]]
        assert used <= v, f"{name}: valence {used} > {v}"
        if atom["el"] in "NO":
            assert used == v, f"{name}: {atom['el']} with {used} bonds (no charges/NH expected)"
        hydrogens[name] = v - used
    log.append("valence ok: " + ", ".join(f"{n}={order[n]}" for n in atoms))

    counts = Counter(a["el"] for a in atoms.values())
    counts["H"] = sum(hydrogens.values())
    formula = "".join(f"{el}{counts[el]}" for el in ("C", "H", "N", "O"))
    assert formula == "C8H10N4O2", formula
    mass = counts["C"] * 12.011 + counts["H"] * 1.008 + counts["N"] * 14.007 + counts["O"] * 15.999
    assert abs(mass - 194.19) < 0.01, mass
    log.append(f"formula {formula}, M = {mass:.2f} g/mol")

    carbonyl = {a for a, b, o in bonds if o == 2 and "O" in (atoms[a]["el"] + atoms[b]["el"])}
    carbonyl = {x for x in carbonyl if atoms[x]["el"] == "C"}
    assert carbonyl == {"C2", "C6"}, carbonyl
    assert bonded("N1", "C2") and bonded("N1", "C6"), "N1 must sit between both carbonyls"
    log.append("N1 bonded to both carbonyl carbons C2 and C6")
    assert bonded("N3", "C2") and bonded("N3", "C4")
    log.append("N3 between C2 and C4")
    hexagon, (hx, hy) = rings["hexagon"]
    pentagon, _ = rings["pentagon"]
    assert "N7" in pentagon and "N7" not in hexagon and bonded("N7", "C5") and bonded("C5", "C6")
    assert atoms["N7"]["pos"][1] < hy and atoms["C6"]["pos"][1] < hy, "N7 must be on the C6=O side"
    log.append("N7 in the 5-ring, on C5, same side as C6=O")
    methyl_on = {n for n in atoms if n.startswith("Me") for n in nbrs[n]}
    assert methyl_on == {"N1", "N3", "N7"}, methyl_on
    assert all(hydrogens[m] == 3 for m in ("Me1", "Me3", "Me7"))
    assert not any(n.startswith("Me") for n in nbrs["N9"]) and hydrogens["N9"] == 0
    log.append("methyls on N1, N3, N7; N9 has no methyl and no H")
    assert hydrogens["C8"] == 1 and order["C8"] == 3
    log.append("C8 carries one H (implicit in the drawing)")
    assert all(hydrogens[c] == 0 for c in ("C2", "C4", "C5", "C6"))

    for a, b, _ in bonds:
        (x1, y1), (x2, y2) = atoms[a]["pos"], atoms[b]["pos"]
        assert abs(math.hypot(x2 - x1, y2 - y1) - L) < 1e-6, f"bond {a}-{b} length"
    for ring, want in ((hexagon, 120.0), (pentagon, 108.0)):
        for i, at in enumerate(ring):
            p, q = ring[i - 1], ring[(i + 1) % len(ring)]
            ax, ay = atoms[at]["pos"]
            v1 = (atoms[p]["pos"][0] - ax, atoms[p]["pos"][1] - ay)
            v2 = (atoms[q]["pos"][0] - ax, atoms[q]["pos"][1] - ay)
            ang = math.degrees(math.acos((v1[0] * v2[0] + v1[1] * v2[1]) / (L * L)))
            assert abs(ang - want) < 1e-6, f"{at}: {ang} != {want}"
    names = list(atoms)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            (x1, y1), (x2, y2) = atoms[a]["pos"], atoms[b]["pos"]
            assert math.hypot(x2 - x1, y2 - y1) > 0.9 * L, f"{a} and {b} too close"
    log.append("all bonds equal length; ring angles 120/108 deg; no crowded atoms")
    return log


def hero_svg(theme: str) -> str:
    DEFS.clear()
    P = PALETTES[theme]
    H = 500
    cx, cy = 400.0, 190.0
    atoms, bonds, rings = build_caffeine(cx, cy)

    label = plex(28)
    sub = plex(19)
    callout = fraunces(29, wght=500, opsz=30)
    num = plex(22)
    word = fraunces(64, wght=420, opsz=96, soft=100, wonk=1)
    tag = plex(20, tracking=0.08)

    out = []
    stroke_w = 3.0
    trim = {"N": 15, "O": 15, "C": 0}

    def labelled(n):
        return atoms[n]["el"] in "NO" or n.startswith("Me")

    def end(a, b):
        """Point on a->b, trimmed if a carries a text label."""
        (x1, y1), (x2, y2) = atoms[a]["pos"], atoms[b]["pos"]
        t = (17 if a.startswith("Me") else trim[atoms[a]["el"]]) / L if labelled(a) else 0
        return x1 + (x2 - x1) * t, y1 + (y2 - y1) * t

    def line(p, q, color=P["ink"], w=stroke_w):
        out.append(
            f'<line x1="{r(p[0])}" y1="{r(p[1])}" x2="{r(q[0])}" y2="{r(q[1])}" '
            f'stroke="{color}" stroke-width="{r(w)}" stroke-linecap="round"/>'
        )

    ring_of = {}
    for members, centre in rings.values():
        for i, m in enumerate(members):
            ring_of[frozenset((m, members[(i + 1) % len(members)]))] = centre
    inner_side = {frozenset(("C4", "C5")): rings["pentagon"][1]}

    for a, b, o in bonds:
        p, q = end(a, b), end(b, a)
        if o == 1:
            line(p, q)
            continue
        dx, dy = q[0] - p[0], q[1] - p[1]
        ln = math.hypot(dx, dy)
        nx, ny = -dy / ln, dx / ln
        key = frozenset((a, b))
        if key in ring_of:  # ring double bond: full line + shorter inner line
            centre = inner_side.get(key, ring_of[key])
            mx, my = (p[0] + q[0]) / 2, (p[1] + q[1]) / 2
            if (centre[0] - mx) * nx + (centre[1] - my) * ny < 0:
                nx, ny = -nx, -ny
            off, sh = 8.0, 0.16
            line(p, q)
            line(
                (p[0] + nx * off + dx * sh, p[1] + ny * off + dy * sh),
                (q[0] + nx * off - dx * sh, q[1] + ny * off - dy * sh),
            )
        else:  # exocyclic C=O: centred pair
            off = 3.8
            line((p[0] + nx * off, p[1] + ny * off), (q[0] + nx * off, q[1] + ny * off))
            line((p[0] - nx * off, p[1] - ny * off), (q[0] - nx * off, q[1] - ny * off))

    def text(t: Text, s, x, y, color, anchor="start"):
        out.append(f'<g fill="{color}">{t.uses(s, x, y, anchor)}</g>')

    cap = 0.70 * 28  # Plex cap height ~0.698 em
    for n, atom in atoms.items():
        x, y = atom["pos"]
        if atom["el"] in "NO" and not n.startswith("Me"):
            text(label, atom["el"], x, y + cap / 2, P["brass"], "middle")

    # methyl groups: the C of the label sits on the atom position
    cw = label.width("C")
    hw = label.width("H")
    sw = sub.width("3")
    methyl_box = {}
    for n in ("Me1", "Me3", "Me7"):
        x, y = atoms[n]["pos"]
        base = y + cap / 2
        if n == "Me1":  # H3C, bond comes in from the right
            x0 = x + cw / 2 - cw - sw - hw
            text(label, "H", x0, base, P["accent"])
            text(sub, "3", x0 + hw, base + 6, P["accent"])
            text(label, "C", x0 + hw + sw, base, P["accent"])
            methyl_box[n] = (x0, x + cw / 2)
        else:  # CH3
            x0 = x - cw / 2
            text(label, "C", x0, base, P["accent"])
            text(label, "H", x0 + cw, base, P["accent"])
            text(sub, "3", x0 + cw + hw, base + 6, P["accent"])
            methyl_box[n] = (x0, x0 + cw + hw + sw)

    # callouts: leader from the methyl label to a numbered tag
    def badge(x, y, n):
        out.append(
            f'<circle cx="{r(x)}" cy="{r(y)}" r="17" fill="none" '
            f'stroke="{P["accent"]}" stroke-width="2"/>'
        )
        text(num, n, x, y + 0.7 * 22 / 2, P["accent"], "middle")

    def leader(p, q):
        out.append(
            f'<line x1="{r(p[0])}" y1="{r(p[1])}" x2="{r(q[0])}" y2="{r(q[1])}" '
            f'stroke="{P["accent"]}" stroke-width="2.6" stroke-dasharray="0.1 5" stroke-linecap="round"/>'
        )

    # each callout: methyl label, dotted leader, numbered badge, role text
    margin = 24
    for me, n, role, side in (("Me1", "1", "HTL Villach", -1), ("Me7", "7", "CTF · AlpSec", 1), ("Me3", "3", "Teledrones", 1)):
        y = atoms[me]["pos"][1]
        edge = methyl_box[me][0] if side < 0 else methyl_box[me][1]
        bx = edge + side * 38
        leader((edge + side * 6, y), (bx - side * 21, y))
        badge(bx, y, n)
        tx = bx + side * 29
        text(callout, role, tx, y + 11, P["ink"], "end" if side < 0 else "start")
        far = tx + side * callout.width(role)
        assert margin <= far <= W - margin, f"callout {n} clipped ({far:.1f})"

    # wordmark
    wy = 410
    text(word, "Methyltheobromin", W / 2, wy, P["ink"], "middle")
    ww = word.width("Methyltheobromin")
    assert ww <= W - 2 * margin, ww
    rule_y = wy + 26
    out.append(
        f'<line x1="{r(W / 2 - ww / 2)}" y1="{rule_y}" x2="{r(W / 2 + ww / 2)}" y2="{rule_y}" '
        f'stroke="{P["brass"]}" stroke-width="1.5"/>'
    )
    text(tag, "1,3,7-TRIMETHYLXANTHIN", W / 2, rule_y + 32, P["muted"], "middle")

    title = "Methyltheobromin: caffeine with its methyl groups on N1, N3 and N7 labelled HTL Villach, Teledrones and CTF · AlpSec"
    return svg_doc(W, H, P["bg"], title, out)


def bg_shape(w, h, bg, corners="all", rad=16) -> str:
    if corners == "all":
        return f'<rect width="{w}" height="{h}" rx="{rad}" fill="{bg}"/>'
    if corners == "none":
        return f'<rect width="{w}" height="{h}" fill="{bg}"/>'
    t = rad if corners == "top" else 0
    b = rad if corners == "bottom" else 0
    d = (f"M0 {t} " + (f"A{t} {t} 0 0 1 {t} 0 " if t else "") + f"H{w - t} "
         + (f"A{t} {t} 0 0 1 {w} {t} " if t else "") + f"V{h - b} "
         + (f"A{b} {b} 0 0 1 {w - b} {h} " if b else "") + f"H{b} "
         + (f"A{b} {b} 0 0 1 0 {h - b} " if b else "") + "Z")
    return f'<path d="{d}" fill="{bg}"/>'


def svg_doc(w, h, bg, title, body, corners="all") -> str:
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}" role="img" aria-label="{title}">',
            f"<title>{title}</title>",
            "<defs>" + "".join(f'<path id="{g}" d="{d}"/>' for g, d in DEFS.values() if d) + "</defs>",
            bg_shape(w, h, bg, corners),
            *body,
            "</svg>",
            "",
        ]
    )


# -------------------------------------------------------------------------- karte


HEAD_H = 120   # header strip height
ROW_H = 100    # height of one project strip
ROW_BASE = 48  # name baseline inside a strip
LAST_PAD = 52  # extra bottom padding on the last strip


def _karte_fonts():
    return dict(
        head=fraunces(50, wght=440, opsz=72, soft=100, wonk=1),
        name=fraunces(34, wght=540, opsz=36, soft=50),
        stack=plex(23),
        blurb=fraunces(27, wght=380, opsz=24, soft=100, italic=True),
    )


def karte_head_svg(theme: str) -> str:
    """Top strip of Die Karte: rule · title · rule."""
    DEFS.clear()
    P = PALETTES[theme]
    F = _karte_fonts()
    margin = 44
    out = []
    hy = 82
    out.append(f'<g fill="{P["ink"]}">{F["head"].uses("Die Karte", W / 2, hy, "middle")}</g>')
    hw = F["head"].width("Die Karte")
    for x1, x2 in ((margin, W / 2 - hw / 2 - 24), (W / 2 + hw / 2 + 24, W - margin)):
        out.append(
            f'<line x1="{r(x1)}" y1="{hy - 14}" x2="{r(x2)}" y2="{hy - 14}" '
            f'stroke="{P["accent"]}" stroke-width="1.5"/>'
        )
    for x in (margin, W - margin):
        out.append(f'<circle cx="{x}" cy="{hy - 14}" r="3.5" fill="{P["accent"]}"/>')
    return svg_doc(W, HEAD_H, P["bg"], "Die Karte", out, corners="all")


def karte_row_svg(theme: str, p: dict, last: bool) -> str:
    """One linked menu row: name ······ stack, blurb underneath."""
    DEFS.clear()
    P = PALETTES[theme]
    F = _karte_fonts()
    margin = 44
    out = []

    def text(t, s, x, y, color, anchor="start"):
        out.append(f'<g fill="{color}">{t.uses(s, x, y, anchor)}</g>')

    y = ROW_BASE
    name, stack, blurb = p["name"], p["stack"], p["blurb"]
    nw, sw = F["name"].width(name), F["stack"].width(stack)
    text(F["name"], name, margin, y, P["ink"])
    text(F["stack"], stack, W - margin, y, P["brass"], "end")
    lx1, lx2 = margin + nw + 14, W - margin - sw - 14
    assert lx2 - lx1 >= 60, f"no room for the leader in {name!r}"
    n = max(int((lx2 - lx1) // 9), 2)
    stepx = (lx2 - lx1) / n
    dots = "".join(f'<circle cx="{r(lx1 + i * stepx)}" cy="{y - 4}" r="1.7"/>' for i in range(n + 1))
    out.append(f'<g fill="{P["brass"]}">{dots}</g>')
    bw = F["blurb"].width(blurb)
    assert bw <= W - 2 * margin - 24, f"blurb too wide for {name!r}: {bw:.0f}"
    text(F["blurb"], blurb, margin + 24, y + 38, P["muted"])
    h = ROW_BASE + 38 + LAST_PAD if last else ROW_H
    title = f"{name} ({stack}): {blurb}"
    return svg_doc(W, h, P["bg"], title, out, corners="all")


def slug(p: dict) -> str:
    return p["repo"].lower()


# --------------------------------------------------------------------------- main


def main() -> int:
    atoms, bonds, rings = build_caffeine(380.0, 190.0)
    for line in check_caffeine(atoms, bonds, rings):
        print("molecule:", line)

    projects = tomllib.loads(DATA.read_text(encoding="utf-8"))["project"]
    assert len(projects) == 6, len(projects)

    ASSETS.mkdir(exist_ok=True)
    for theme in PALETTES:
        files = [("hero", hero_svg(theme)), ("karte-head", karte_head_svg(theme))]
        files += [
            (f"karte-{slug(p)}", karte_row_svg(theme, p, i == len(projects) - 1))
            for i, p in enumerate(projects)
        ]
        for stem, svg in files:
            path = ASSETS / f"{stem}-{theme}.svg"
            path.write_text(svg, encoding="utf-8")
            print(f"wrote {path.relative_to(ROOT)} ({len(svg.encode()) // 1024} KiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
