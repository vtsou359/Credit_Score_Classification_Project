"""ScoreBridge brand primitives: palette + the arch-gauge mark as SVG."""
import math

# Brand colour: crimson, the middle step of the class palette used in custom_functions/utility_functions.py
CORAL, CRIMSON, WINE = '#F05941', '#BE3144', '#872341'
INK, PAPER = '#0F1118', '#FAF7F5'

WHITE_RAMP = ('rgba(255,255,255,0.42)', 'rgba(255,255,255,0.68)', '#FFFFFF')
TILE_GRADIENT = ('#CF3A4C', '#A82B3E')  # subtle diagonal around CRIMSON


def _pt(cx, cy, r, deg):
    a = math.radians(deg)
    return cx + r * math.cos(a), cy - r * math.sin(a)


def _arc(cx, cy, r, a0, a1):
    x0, y0 = _pt(cx, cy, r, a0)
    x1, y1 = _pt(cx, cy, r, a1)
    return f'M{x0:.2f},{y0:.2f} A{r},{r} 0 0 1 {x1:.2f},{y1:.2f}'


def mark(colors=(WINE, CRIMSON, CORAL), needle='#FFFFFF', cx=50, cy=64, r=30, w=9.5, gap=26,
         needle_deg=40, needle_len=21, needle_w=4.6, pivot=6):
    """Three-segment arch gauge (Poor -> Standard -> Good, left to right) with the needle on 'Good'.
    Returns SVG elements for a 0 0 100 100 viewBox."""
    span = (180 - 2 * gap) / 3
    parts, a = [], 180
    for c in colors:
        parts.append(f'<path d="{_arc(cx, cy, r, a, a - span)}" stroke="{c}" stroke-width="{w}" '
                     f'fill="none" stroke-linecap="round"/>')
        a -= span + gap
    x, y = _pt(cx, cy, needle_len, needle_deg)
    parts.append(f'<line x1="{cx}" y1="{cy}" x2="{x:.2f}" y2="{y:.2f}" stroke="{needle}" '
                 f'stroke-width="{needle_w}" stroke-linecap="round"/>')
    parts.append(f'<circle cx="{cx}" cy="{cy}" r="{pivot}" fill="{needle}"/>')
    return ''.join(parts)


def _gradient(uid):
    return (f'<linearGradient id="{uid}g" x1="0" y1="0" x2="1" y2="1">'
            f'<stop offset="0" stop-color="{TILE_GRADIENT[0]}"/><stop offset="1" stop-color="{TILE_GRADIENT[1]}"/>'
            f'</linearGradient>')


def tile_svg(size=None, radius=22, uid='t'):
    """App-icon style logo: white gauge on a rounded crimson tile."""
    dims = f' width="{size}" height="{size}"' if size else ''
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"{dims}>'
            f'<defs>{_gradient(uid)}</defs>'
            f'<rect width="100" height="100" rx="{radius}" fill="url(#{uid}g)"/>'
            f'{mark(colors=WHITE_RAMP)}</svg>')


def square_svg(size=None, uid='s'):
    """Full-bleed square logo (safe for LinkedIn media, which may not keep transparency)."""
    dims = f' width="{size}" height="{size}"' if size else ''
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"{dims}>'
            f'<defs>{_gradient(uid)}</defs><rect width="100" height="100" fill="url(#{uid}g)"/>'
            f'<g transform="translate(50 50) scale(0.86) translate(-50 -49)">{mark(colors=WHITE_RAMP)}</g></svg>')
