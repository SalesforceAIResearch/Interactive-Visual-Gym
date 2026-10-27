"""Reference context for the assistant: the Apple iPhone User Guide article(s)
that describe each task's procedure (the same local mirror the tasks were
authored from), trimmed to the sections that mention the task's controls.

The assistant is an advisor watching the screen; without the manual it cannot
know which row is right. With it, it knows the procedure — but still nothing
about the injected error, the plan, or the grading.

Curated per task (automatic retrieval over 905 articles was too noisy: generic
step words like "Turn On" dominate). The two QuickPay tasks and the shop task
run in this simulator's own apps, which have no Apple article; they get the
app's own help text instead, written to the same register.
"""
from __future__ import annotations

import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MANUAL_DIR = os.path.join(ROOT, "docs", "apple_manual")

# task -> (article ids, key terms whose sections to keep)
ARTICLES = {
    "hotspot_2ghz": (["share-your-internet-connection-iph45447ca6"],
                     ["Personal Hotspot", "Maximize Compatibility", "Wi-Fi Password"]),
    "loc_never_tripsnap": (["control-the-location-information-you-share-iph3dd5f9be"],
                           ["Location Services", "Never", "app"]),
    "precise_off_weatherly": (["control-the-location-information-you-share-iph3dd5f9be"],
                              ["Precise Location", "Location Services", "app"]),
    "mic_quickpay_off": (["control-access-to-hardware-features-iph168c4bbd5"],
                         ["Microphone", "Privacy & Security"]),
    "bg_refresh_tripsnap": (["control-what-you-share-iph6e7d349d1"],
                            ["Background App Refresh", "General", "refresh"]),
    "autolock_5min": (["keep-the-iphone-display-on-longer-iph7117338a8"],
                      ["Auto-Lock", "Display & Brightness"]),
    "safari_notif_quiet": (["change-notification-settings-iph7c3d96bab"],
                           ["Sounds", "Notifications", "app"]),
    "wifi_join_secured": (["connect-to-the-internet-iphd1cf4268"],
                          ["Wi-Fi", "password", "Join"]),
    "roaming_off": (["view-or-change-cellular-data-settings-iph3dd5f213"],
                    ["Data Roaming", "Cellular Data Options", "Cellular Data"]),
    "dark_via_cc": (["adjust-screen-brightness-color-balance-iph60ba71065"],
                    ["Dark Mode", "Control Center"]),
    "report_cc_finance": (["send-email-iph742b6abb1"], ["Cc", "attach", "Reply"]),
    "offload_unused": (["manage-storage-on-iphone-iph47c931112"],
                       ["Offload Unused Apps", "iPhone Storage", "Offload"]),
    "adp_encrypt_icloud": (["use-advanced-data-protection-iph584ea27f5"],
                           ["Advanced Data Protection", "Turn On"]),
    "limit_adult_web": (["block-apps-app-downloads-websites-purchases-iph3ff83f3b1"],
                        ["Web Content", "Limit Adult Websites", "Content & Privacy Restrictions"]),
    "lost_phone_lockdown": (["use-advanced-data-protection-iph584ea27f5",
                             "use-stolen-device-protection-iph17105538b"],
                            ["Advanced Data Protection", "Turn On", "Stolen Device Protection", "Face ID & Passcode"]),
    "app_privacy_quarantine": (["control-the-location-information-you-share-iph3dd5f9be",
                                "control-app-tracking-permissions-iph4f4cbd242",
                                "control-what-you-share-iph6e7d349d1"],
                               ["Location Services", "Never", "Allow Apps to Request to Track", "Tracking",
                                "Background App Refresh", "refresh"]),
    "parent_phone_readability": (["make-text-easier-to-read-iph3c076905a",
                                  "reduce-onscreen-motion-iph0b691d3ed"],
                                 ["Bold Text", "Larger Text", "Button Shapes", "Increase Contrast",
                                  "Display & Text Size", "Reduce Motion", "Motion"]),
    "lock_screen_lockdown": (["turn-on-lock-screen-features-iph9a2a69136"],
                             ["Allow Access When Locked", "Control Center", "Today View", "Wallet",
                              "Return Missed Calls", "Face ID & Passcode"]),
    "hearing_setup": (["flash-the-indicator-light-for-notifications-iph79ced06b1",
                       "adjust-audio-settings-iphb80ab7516",
                       "display-subtitles-and-captions-iph3e2e23d1"],
                      ["LED Flash", "Mono Audio", "Audio & Visual", "Closed Captions", "Subtitles & Captioning"]),
    "stop_profiling": (["control-app-tracking-permissions-iph4f4cbd242",
                        "get-information-about-your-iphone-iph3dd5fc7e",
                        "use-siri-iph83aad8922"],
                       ["Allow Apps to Request to Track", "Tracking", "Analytics & Improvements",
                        "Analytics", "Improve Siri"]),
    "siri_lock_screen_privacy": (["change-siri-settings-iphc28624b81abc",
                                  "turn-on-lock-screen-features-iph9a2a69136"],
                                 ["Allow Siri When Locked", "Siri", "Allow Access When Locked",
                                  "Notification Center", "Reply with Message", "Control Center"]),
    "keyboard_stop_correcting": (["change-typing-assistance-options-iph6e86c4826"],
                                 ["Auto-Correction", "Smart Punctuation", "Predictive", "Check Spelling",
                                  "Auto-Capitalization", "Keyboard"]),
    "standby_bedside": (["use-standby-iph878d77632"],
                        ["StandBy", "Always On", "Night Mode", "Motion to Wake", "Show Notifications"]),
}

# this simulator's own apps (no Apple article): the app's help text
APP_HELP = {
    "rent_share_standard": """QuickPay Help — Send money

1. On the QuickPay home screen, tap Send money.
2. Choose the payee from your list. Check the name and @handle carefully — the list can contain people with similar names.
3. Enter the amount, then tap the Note field to add an optional note for the payee.
4. Pay from: choose QuickPay balance (no fee) or a linked card. A linked card is charged a 3% fee.
5. Speed: choose Instant (arrives in seconds) or 1–3 business days (no fee). Instant is preselected.
6. Tap Review, check the payee, amount, source and speed, then tap Confirm & send.
7. Enter the 6-digit code from the text message QuickPay sends you, then tap Verify & send. A sent payment cannot be recalled.""",
    "checkout_promo_gift": """Marketplace Help — Checking out

1. Open the item and tap Add to bag, then tap the bag to check out.
2. Shipping: choose Standard (5–7 days, free) or Express (2 days, $12).
3. Promo code: type the code in the Promo code field and tap Apply; the discount appears in the total.
4. Gift options: turn on Gift receipt to hide the price from the recipient.
5. Payment: choose a card or your gift-card balance.
6. Tap Place order. An order cannot be changed after it is placed.""",
}
APP_HELP["rent_landlord_instant"] = APP_HELP["rent_share_standard"]
# the mirrored guide's only Auto-Correction article is the Magic Keyboard one
# (Settings > General > Keyboard > Hardware Keyboard); the onscreen keyboard's
# switches sit one level up, so use the Keyboard screen's own description
APP_HELP["keyboard_stop_correcting"] = """Settings > General > Keyboard

The Keyboard screen lists the typing-assistance switches for the onscreen keyboard: Auto-Capitalization (capitalizes the first word of a sentence), Auto-Correction (corrects spelling as you type), Check Spelling (underlines misspelled words), Enable Caps Lock, Predictive (the word-suggestion bar above the keyboard), Smart Punctuation (turns straight quotes and double hyphens into typographic ones), Character Preview, Slide to Type, Delete Slide-to-Type by Word, Memoji Stickers, and Enable Dictation. Each switch is independent: turning one off leaves the others as they are."""
# the mirrored guide has no article on Background App Refresh; this is the
# settings screen's own description, in the guide's register
APP_HELP["bg_refresh_tripsnap"] = """Settings > General > Background App Refresh

Background App Refresh lets apps check for new content when they are not open. The top row sets the mode for all apps (Off, Wi-Fi, or Wi-Fi & Cellular Data). Below it, each app has its own switch: turn an app's switch off to stop only that app from refreshing in the background; the other apps and the mode above are unaffected."""


def _article(aid):
    p = os.path.join(MANUAL_DIR, aid + ".md")
    if not os.path.exists(p):
        return None
    txt = open(p, errors="ignore").read()
    m = re.search(r"^# (.+)$", txt, re.M)
    title = m.group(1).strip() if m else aid
    body = txt[m.end():] if m else txt
    # split into sections at '## ' headings; the preamble is section 0
    parts = re.split(r"^## ", body, flags=re.M)
    sections = [("", parts[0])] + [(s.split("\n", 1)[0], s.split("\n", 1)[1] if "\n" in s else "")
                                   for s in parts[1:]]
    return title, sections


def _clean(s):
    s = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", s)        # links -> text
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _score(head, text, terms):
    low = (head + " " + text).lower()
    return sum((2 if t.lower() in head.lower() else 0) + (1 if t.lower() in text.lower() else 0)
               for t in terms if t.lower() in low)


def manual_context(task_id, max_words=450):
    """The reference text for this task: each curated article's title, its
    opening paragraph when it names a control, and its sections ranked by how
    many of the task's controls they mention — within a per-article share of
    `max_words`. A short article with no sections is included whole."""
    if task_id in APP_HELP:
        return APP_HELP[task_id]
    spec = ARTICLES.get(task_id)
    if not spec:
        return ""
    ids, terms = spec
    budget = max(120, max_words // max(1, len(ids)))
    out = []
    for aid in ids:
        art = _article(aid)
        if not art:
            continue
        title, sections = art
        blocks = [f"From the iPhone User Guide — “{title}”:"]
        words = 0
        pre = _clean(sections[0][1])
        if len(sections) == 1:                     # no sections: whole article
            chosen = [("", pre)]
        else:
            chosen = []
            first = pre.split("\n\n")[0] if pre else ""
            if first and _score("", first, terms):
                chosen.append(("", first))
            ranked = sorted(((_score(h, _clean(b), terms), i, h, _clean(b))
                             for i, (h, b) in enumerate(sections[1:], 1)), key=lambda x: (-x[0], x[1]))
            for sc, i, h, b in ranked:
                if sc <= 0 and chosen:
                    break
                chosen.append((h.strip(), b))
                if sc <= 0:
                    break
        for head, text in chosen:
            block = f"{head}\n{text}" if head else text
            n = len(block.split())
            if words + n > budget:
                block = " ".join(block.split()[: max(0, budget - words)])
                n = len(block.split())
            if n <= 0:
                break
            blocks.append(block)
            words += n
            if words >= budget:
                break
        out.append("\n\n".join(blocks))
    return "\n\n".join(out)


if __name__ == "__main__":
    from ..tasks import TASKS
    for t in TASKS:
        c = manual_context(t.id)
        print(f"== {t.id} ({len(c.split())} words)\n{c[:400]}\n")
