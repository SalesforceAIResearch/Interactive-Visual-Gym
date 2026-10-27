"""Inline SVG icons + iOS colored-square tints.

SF Symbols use private-use codepoints only present in Apple's SF Pro font, which
we can't ship. Instead we draw a compact set of white monochrome SVG glyphs and
place them on colored rounded squares — the iconic iOS Settings look.
"""

# each entry: SVG inner markup drawn white on a 24x24 viewBox
_P = {
    "wifi": '<path d="M12 18.5a1.4 1.4 0 100-2.8 1.4 1.4 0 000 2.8z" fill="#fff"/>'
            '<path d="M5 11a10 10 0 0114 0" stroke="#fff" stroke-width="2" fill="none" stroke-linecap="round"/>'
            '<path d="M8 14a6 6 0 018 0" stroke="#fff" stroke-width="2" fill="none" stroke-linecap="round"/>',
    "bt": '<path d="M8 7l8 5-4 3 4 3-8 5V4l4 3-4 3" stroke="#fff" stroke-width="1.7" '
          'fill="none" stroke-linejoin="round" stroke-linecap="round"/>',
    "apps": '<g fill="#fff"><rect x="4" y="4" width="6" height="6" rx="1.5"/>'
            '<rect x="14" y="4" width="6" height="6" rx="1.5"/>'
            '<rect x="4" y="14" width="6" height="6" rx="1.5"/>'
            '<rect x="14" y="14" width="6" height="6" rx="1.5"/></g>',
    "bell": '<path d="M12 4a5 5 0 015 5v4l2 3H5l2-3V9a5 5 0 015-5z" fill="#fff"/>'
            '<path d="M10 20a2 2 0 004 0" stroke="#fff" stroke-width="1.6" fill="none"/>',
    "sun": '<circle cx="12" cy="12" r="4" fill="#fff"/>'
           '<g stroke="#fff" stroke-width="1.8" stroke-linecap="round">'
           '<path d="M12 3v2M12 19v2M3 12h2M19 12h2M5.6 5.6l1.4 1.4M17 17l1.4 1.4M18.4 5.6L17 7M7 17l-1.4 1.4"/></g>',
    "vol": '<path d="M4 9h3l4-4v14l-4-4H4z" fill="#fff"/>'
           '<path d="M15 8a5 5 0 010 8M17.5 6a8 8 0 010 12" stroke="#fff" stroke-width="1.7" fill="none" stroke-linecap="round"/>',
    "disk": '<g fill="#fff"><rect x="4" y="5" width="16" height="5" rx="1.5"/>'
            '<rect x="4" y="13" width="16" height="5" rx="1.5"/></g>'
            '<circle cx="8" cy="7.5" r="1" fill="#8e8e93"/><circle cx="8" cy="15.5" r="1" fill="#8e8e93"/>',
    "batt": '<rect x="3" y="8" width="16" height="9" rx="2.5" fill="#fff"/>'
            '<rect x="20" y="10.5" width="2" height="4" rx="1" fill="#fff"/>',
    "hotspot": '<circle cx="12" cy="15" r="2.2" fill="#fff"/>'
               '<path d="M8 11a5.5 5.5 0 018 0M5.5 8.5a9 9 0 0113 0" stroke="#fff" stroke-width="1.8" fill="none" stroke-linecap="round"/>',
    "airplane": '<path d="M12 3l2 7 6 4v2l-6-1.5V19l2 1.5V22l-4-1-4 1v-1.5L8 19v-4.5L2 16v-2l6-4 2-7z" fill="#fff"/>',
    "cell": '<g fill="#fff"><rect x="3" y="14" width="3" height="6" rx="1"/>'
            '<rect x="8" y="10" width="3" height="10" rx="1"/>'
            '<rect x="13" y="6" width="3" height="14" rx="1"/>'
            '<rect x="18" y="3" width="3" height="17" rx="1"/></g>',
    "person": '<circle cx="12" cy="8" r="4" fill="#fff"/>'
              '<path d="M4 20a8 8 0 0116 0z" fill="#fff"/>',
    "camera": '<rect x="3" y="7" width="18" height="12" rx="3" fill="#fff"/>'
              '<circle cx="12" cy="13" r="3.2" fill="#8e8e93"/>'
              '<rect x="8" y="4" width="8" height="4" rx="1.5" fill="#fff"/>',
    "mic": '<rect x="9" y="3" width="6" height="11" rx="3" fill="#fff"/>'
           '<path d="M6 11a6 6 0 0012 0M12 17v4" stroke="#fff" stroke-width="1.8" fill="none" stroke-linecap="round"/>',
    "pin": '<path d="M12 2a7 7 0 017 7c0 5-7 13-7 13S5 14 5 9a7 7 0 017-7z" fill="#fff"/>'
           '<circle cx="12" cy="9" r="2.6" fill="#ff3b30"/>',
    "vibrate": '<rect x="8" y="4" width="8" height="16" rx="2.5" fill="#fff"/>'
               '<path d="M4 8v8M20 8v8" stroke="#fff" stroke-width="1.8" stroke-linecap="round"/>',
    "darkmode": '<circle cx="12" cy="12" r="9" fill="none" stroke="#fff" stroke-width="1.8"/>'
                '<path d="M12 3a9 9 0 000 18z" fill="#fff"/>',
    "moon": '<path d="M15 3a9 9 0 106 15A7 7 0 0115 3z" fill="#fff"/>',
    "lock": '<rect x="5" y="10" width="14" height="10" rx="2.5" fill="#fff"/>'
            '<path d="M8 10V8a4 4 0 018 0v2" stroke="#fff" stroke-width="1.9" fill="none"/>',
    "pay": '<rect x="3" y="6" width="18" height="12" rx="2.5" fill="#fff"/>'
           '<rect x="3" y="9" width="18" height="2.5" fill="#8e8e93"/>',
    "privacy": '<path d="M12 3c-1.4 0-2 .9-2 2v5H9V6.5a1.3 1.3 0 00-2.6 0V13c0 .6-.7.8-1 .3l-.8-1.2a1.3 1.3 0 00-2.2 1.4l2.4 4.3A5 5 0 0011 20h1.5a5.5 5.5 0 005.5-5.5V8a1.3 1.3 0 00-2.6 0v2h-1V6a1.3 1.3 0 00-2.6 0v4h-1V5c0-1.1-.6-2-1.8-2z" fill="#fff"/>',
    "vpn": '<path d="M12 3l7 3v5c0 5-3.5 8-7 10-3.5-2-7-5-7-10V6z" fill="#fff"/>',
    # app-launcher glyphs
    "gear": '<path d="M19.4 12c0-.5-.05-1-.13-1.5l2-1.6-2-3.4-2.4 1a7.5 7.5 0 00-2.6-1.5L13.9 2h-3.8l-.37 2.5A7.5 7.5 0 007.1 6l-2.4-1-2 3.4 2 1.6a7.6 7.6 0 000 3l-2 1.6 2 3.4 2.4-1a7.5 7.5 0 002.6 1.5l.37 2.5h3.8l.37-2.5a7.5 7.5 0 002.6-1.5l2.4 1 2-3.4-2-1.6c.08-.5.13-1 .13-1.5z" fill="#fff"/>'
            '<circle cx="12" cy="12" r="3.4" fill="#8e8e93"/>',
    "chat": '<path d="M4 5h16a1 1 0 011 1v9a1 1 0 01-1 1H9l-4 4v-4H4a1 1 0 01-1-1V6a1 1 0 011-1z" fill="#fff"/>',
    "cart": '<path d="M6 7h13l-1.5 8H8L6 4H3" stroke="#fff" stroke-width="1.9" fill="none" '
            'stroke-linecap="round" stroke-linejoin="round"/>'
            '<circle cx="9" cy="19" r="1.5" fill="#fff"/><circle cx="16" cy="19" r="1.5" fill="#fff"/>',
    "mail": '<rect x="3" y="6" width="18" height="12" rx="2.5" fill="#fff"/>'
            '<path d="M4 8l8 5 8-5" stroke="#8e8e93" stroke-width="1.6" fill="none"/>',
    "phone": '<path d="M7 3c1 0 1.6.4 2 1.5l1 2.6c.3.9.1 1.5-.6 2l-1 .8c1 2 2.4 3.4 4.4 4.4l.8-1c.5-.7 1.1-.9 2-.6l2.6 1c1.1.4 1.5 1 1.5 2 0 3.2-2.6 4.8-5 4.3C9.4 21.6 2.4 14.6 2.7 8 2.9 5.6 4 3 7 3z" fill="#fff"/>',
    "photo": '<rect x="3" y="4" width="18" height="16" rx="3" fill="#fff"/>'
             '<circle cx="8.5" cy="9.5" r="1.8" fill="#ff9500"/>'
             '<path d="M4 18l5-5 3 3 4-4 4 4v3a1 1 0 01-1 1H5a1 1 0 01-1-1z" fill="#34c759"/>',
    "share": '<path d="M12 3l4 4M12 3L8 7M12 3v12" stroke="#007aff" stroke-width="2" '
             'fill="none" stroke-linecap="round" stroke-linejoin="round"/>'
             '<path d="M6 11H5a1 1 0 00-1 1v8a1 1 0 001 1h14a1 1 0 001-1v-8a1 1 0 00-1-1h-1" '
             'stroke="#007aff" stroke-width="2" fill="none" stroke-linecap="round"/>',
    "replyarrow": '<path d="M9 5L3 11l6 6M3.5 11H15a5 5 0 015 5v3" stroke="#007aff" '
                  'stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"/>',
    "flag": '<path d="M6 3v18M6 4h11l-2 4 2 4H6" stroke="#007aff" stroke-width="2" '
            'fill="none" stroke-linecap="round" stroke-linejoin="round"/>',
    "sendarrow": '<circle cx="12" cy="12" r="11" fill="#007aff"/>'
                 '<path d="M12 17V7M12 7l-4.5 4.5M12 7l4.5 4.5" stroke="#fff" '
                 'stroke-width="2.2" fill="none" stroke-linecap="round" '
                 'stroke-linejoin="round"/>',
    "trash": '<path d="M5 7h14M9 7V5a1 1 0 011-1h4a1 1 0 011 1v2m-9 0l1 13a1 1 0 001 1h6a1 1 0 001-1l1-13" '
             'stroke="#007aff" stroke-width="1.9" fill="none" stroke-linecap="round" stroke-linejoin="round"/>',
}

# iOS-style tint per icon (the colored square behind the white glyph)
TINT = {
    "wifi": "#2f7bf6", "bt": "#2f7bf6", "cell": "#34c759", "apps": "#8e8e93",
    "bell": "#ff3b30", "sun": "#5aa9ff", "moon": "#1c1c1e", "vol": "#ff3b30",
    "disk": "#8e8e93", "batt": "#34c759", "hotspot": "#34c759", "airplane": "#ff9500",
    "person": "#ff9500", "camera": "#8e8e93", "mic": "#ff9500", "pin": "#2f7bf6",
    "vibrate": "#ff2d55", "lock": "#8e8e93", "pay": "#34c759", "vpn": "#2f7bf6",
    "privacy": "#2f7bf6",
}


def svg_icon(name, size=18):
    inner = _P.get(name)
    if not inner:
        return ""
    return (f'<svg viewBox="0 0 24 24" width="{size}" height="{size}" '
            f'xmlns="http://www.w3.org/2000/svg">{inner}</svg>')


def has(name):
    return name in _P


# ---- status-bar glyphs (dark, not on a square) ----
def status_signal(color="#000"):
    return (f'<svg viewBox="0 0 24 14" width="18" height="12" xmlns="http://www.w3.org/2000/svg">'
            f'<g fill="{color}"><rect x="0" y="9" width="3.5" height="5" rx="1"/>'
            f'<rect x="6" y="6" width="3.5" height="8" rx="1"/>'
            f'<rect x="12" y="3" width="3.5" height="11" rx="1"/>'
            f'<rect x="18" y="0" width="3.5" height="14" rx="1"/></g></svg>')


def status_wifi(color="#000"):
    return (f'<svg viewBox="0 0 20 15" width="17" height="13" xmlns="http://www.w3.org/2000/svg">'
            f'<path d="M10 13.5a1.4 1.4 0 100-2.8 1.4 1.4 0 000 2.8z" fill="{color}"/>'
            f'<path d="M3.5 6a10 10 0 0113 0" stroke="{color}" stroke-width="1.9" fill="none" stroke-linecap="round"/>'
            f'<path d="M6 8.5a6 6 0 018 0" stroke="{color}" stroke-width="1.9" fill="none" stroke-linecap="round"/></svg>')


def search_glyph(color="#8a8a8e"):
    return (f'<svg viewBox="0 0 24 24" width="15" height="15" xmlns="http://www.w3.org/2000/svg">'
            f'<circle cx="10" cy="10" r="6" stroke="{color}" stroke-width="2" fill="none"/>'
            f'<path d="M15 15l5 5" stroke="{color}" stroke-width="2" stroke-linecap="round"/></svg>')
