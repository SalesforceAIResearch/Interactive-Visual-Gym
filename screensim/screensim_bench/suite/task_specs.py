#!/usr/bin/env python3
"""Hand-written task specs for the RL library. THIS is the file written by hand.

One dict per task. `steps` lists (page key, actions on that page) in the order the owner would do
them; the compiler derives the route, the full gesture plan and the goal state, then verifies both
by replay (screensim_bench/suite/build_tasks.py). Keep `instruction` OUTCOME-worded: say what the owner wants to
be true, never which switch to press — the assistant is supposed to work that out.

Bands: 1 one page, a couple of controls · 2 one page with a leave-alone constraint ·
3 benchmark parity (several pages and/or a real trap) · 4 above benchmark (6+ changes, 3+ pages,
at least one value choice).
"""

SPECS = [
    # ---------------------------------------------------------------- band 1
    dict(id="camera_framing_guides", family="camera", band=1,
         instruction="My photos come out crooked and my selfies come out flipped the wrong way "
                     "round from what I saw while taking them. Put the lines across the viewfinder "
                     "that split it into thirds so I can line a shot up, and make the front camera "
                     "save selfies the way they looked on screen. Leave the rest of the camera alone.",
         trap="'Level' is already on and looks like the framing aid; 'View Outside the Frame' is not it",
         mirrors="Apple Support: Take photos with the grid; Mirror front camera",
         steps=[("settings.stub_camera", ["tap Grid", "tap Mirror Front Camera"])]),

    dict(id="motion_calm_basic", family="accessibility", band=1,
         instruction="Sliding animations and anything that strobes set off my migraines. Cut the "
                     "movement in the interface right down, dim the flashing, and hold the screen "
                     "refresh at the lower rate. I still want message effects and video previews to "
                     "play by themselves — don't touch those.",
         trap="the auto-play switches sit in the same list and look like 'motion' too",
         mirrors="Apple Support: Reduce screen motion",
         steps=[("settings.acc_motion", ["tap Reduce Motion", "tap Dim Flashing Lights",
                                         "tap Limit Frame Rate"])]),

    dict(id="readability_basic", family="accessibility", band=1,
         instruction="My eyes are getting worse. Make the text heavier, push the contrast up, and "
                     "put the little on/off markers inside the switches so I can tell at a glance "
                     "which way each one is set. Don't invert the colours — I can't stand how that looks.",
         trap="Smart Invert and Classic Invert are the tempting 'contrast' answers and break the request",
         mirrors="Apple Support: Display & Text Size accommodations",
         steps=[("settings.acc_display", ["tap Bold Text", "tap Increase Contrast",
                                          "tap On/Off Labels"])]),

    # ---------------------------------------------------------------- band 2
    dict(id="autoplay_stop", family="accessibility", band=2,
         instruction="Every time I open a message the effects fly around the screen, and videos and "
                     "animated pictures start playing on their own. Stop all three of those from "
                     "playing by themselves. Don't slow the whole interface down though — the "
                     "general motion setting stays as it is.",
         trap="Reduce Motion looks like the one-switch answer and is explicitly not wanted",
         mirrors="Apple Support: Auto-play settings under Motion",
         steps=[("settings.acc_motion", ["tap Auto-Play Message Effects",
                                         "tap Auto-Play Video Previews",
                                         "tap Auto-Play Animated Images"])]),

    dict(id="child_allowed_apps", family="screen_time", band=2,
         instruction="I'm handing this phone to my nephew for the afternoon. He shouldn't be able to "
                     "get on the web, take photos, or fling files to strangers standing nearby. He "
                     "does need to be able to reach me, so leave email and video calling working.",
         trap="turning off Mail or FaceTime breaks the one thing he must keep",
         mirrors="Apple Support: Content & Privacy Restrictions — allowed apps",
         steps=[("settings.gen_allowed_apps_features", ["tap Safari", "tap Camera", "tap AirDrop"])]),

    dict(id="lockscreen_private", family="privacy", band=2,
         instruction="Anyone who picks my phone up can read my notifications and search through my "
                     "stuff without unlocking it. Shut both of those off at the lock screen. The "
                     "quick-settings panel has to keep working there — that's my flashlight.",
         trap="Control Center is the switch people grab first and it must stay on",
         mirrors="Apple Support: Allow access when locked",
         steps=[("settings.lockaccess", ["tap Notification Center", "tap Today View and Search"])]),

    # ---------------------------------------------------------------- band 3
    dict(id="stranger_shield", family="privacy", band=3,
         instruction="I'm drowning in spam. Calls from numbers that aren't in my contacts shouldn't "
                     "ring at all, the same for video calls, and texts from people I don't know "
                     "should be sorted out of my main list instead of landing in it.",
         trap="the three switches live in three different app pages; blocking a contact does nothing here",
         mirrors="Apple Support: Silence unknown callers; Filter unknown senders",
         steps=[("settings.app_Phone", ["tap Silence Unknown Callers"]),
                ("settings.app_FaceTime", ["tap Silence Unknown Callers"]),
                ("settings.app_Messages", ["tap Filter Unknown Senders"])]),

    dict(id="travel_data_diet", family="cellular", band=3,
         instruction="I land in Lisbon tomorrow and data there costs a fortune. Only the browser and "
                     "the maps app may use mobile data — everything else waits for wi-fi. I still "
                     "need mobile data itself switched on for those two.",
         trap="switching off Cellular Data at the top kills the two apps that must keep working",
         mirrors="Apple Support: Set cellular data options per app",
         steps=[("settings.cellular", ["tap TripSnap", "tap QuickPay", "tap Weatherly", "tap Music"])]),

    dict(id="motion_calm_full", family="accessibility", band=3,
         instruction="Since my concussion, movement on the screen makes me nauseous. Take the motion "
                     "out of the interface, stop anything that plays or animates on its own, and dim "
                     "whatever flashes. Leave the screen refresh rate alone — that one doesn't bother me.",
         trap="Limit Frame Rate is in the same list and is explicitly not wanted",
         mirrors="Apple Support: Reduce screen motion; auto-play settings",
         steps=[("settings.acc_motion", ["tap Reduce Motion", "tap Auto-Play Message Effects",
                                         "tap Auto-Play Video Previews", "tap Auto-Play Animated Images",
                                         "tap Dim Flashing Lights"])]),

    # ---------------------------------------------------------------- band 4
    dict(id="grandparent_setup_full", family="accessibility", band=4,
         instruction="I'm setting this phone up for my grandmother before she flies out. She can't "
                     "read thin grey text, she can't tell which switches are on, she can't see where "
                     "a button ends and the background starts, she gets lost when things slide "
                     "around, and the screen keeps locking while she's still reading it — give her "
                     "five minutes before it does. Don't invert the colours, and don't stop it "
                     "locking altogether.",
         trap="Smart Invert answers 'can't read it'; 'Never' answers the locking complaint and leaves "
              "the phone open forever",
         mirrors="Apple Support: Display & Text Size; Reduce Motion; Auto-Lock",
         steps=[("settings.acc_display", ["tap Bold Text", "tap Increase Contrast",
                                          "tap On/Off Labels", "tap Button Shapes"]),
                ("settings.acc_motion", ["tap Reduce Motion"]),
                ("settings.display", ["tap Auto-Lock", "tap 5 Minutes"])]),

    dict(id="lost_phone_drill", family="security", band=4,
         instruction="I'm going somewhere with a lot of pickpockets. If someone lifts this phone, "
                     "they shouldn't be able to read my notifications, answer my messages, pay with "
                     "my cards or call anyone back — none of that from the locked screen. It should "
                     "want the passcode the instant the screen goes dark, and it should wipe itself "
                     "if someone sits there guessing the passcode. Face ID still has to unlock it for me.",
         trap="turning iPhone Unlock off locks the owner out and is the tempting 'make it secure' move",
         mirrors="Apple Support: Allow access when locked; Require passcode; Erase data",
         steps=[("settings.lockaccess", ["tap Notification Center", "tap Reply with Message",
                                         "tap Wallet", "tap Return Missed Calls"]),
                ("settings.stub_faceid", ["tap Require Passcode", "tap Immediately"]),
                ("settings.stub_faceid", ["tap Erase Data"])]),

    # ================================================================ batch 2
    # ---------------------------------------------------------------- band 1
    dict(id="mail_thread_tidy", family="mail", band=1,
         instruction="When I open my mail I can't tell who else was on a message, and replies get "
                     "bundled up so I lose track of them. Show me who a message went to besides me, "
                     "and stop grouping the replies together.",
         trap="'Ask Before Deleting' is the other switch on that page and has nothing to do with it",
         mirrors="Apple Support: Mail settings — To/Cc labels, threading",
         steps=[("settings.app_Mail", ["tap Show To/Cc Labels", "tap Organize by Thread"])]),

    dict(id="tripsnap_permission_trim", family="privacy", band=1,
         instruction="TripSnap is a travel diary — it has no business with my camera or my address "
                     "book. Cut both of those off. It may keep everything else it has.",
         trap="the app's microphone switch is already off; turning off its data or refresh is not asked for",
         mirrors="Apple Support: Control app access to camera and contacts",
         steps=[("settings.app_TripSnap", ["tap Camera", "tap Contacts"])]),

    dict(id="meeting_manners", family="sounds", band=1,
         instruction="In meetings I want the phone to buzz even when it's on silent, and I want it "
                     "to stop clicking at everyone every time I type.",
         trap="'Vibrate on Ring' is already on and is the other, wrong, vibration switch",
         mirrors="Apple Support: Sounds & Haptics",
         steps=[("settings.sounds", ["tap Vibrate on Silent", "tap Keyboard Clicks"])]),

    dict(id="sos_discreet", family="safety", band=1,
         instruction="If I'm ever in trouble I want to call the emergency services by pressing the "
                     "side button five times — and I don't want the phone announcing it with a "
                     "countdown noise while I'm doing it.",
         trap="'Call with Hold and Release' is already on and is the other way of triggering the call",
         mirrors="Apple Support: Emergency SOS",
         steps=[("settings.stub_sos", ["tap Call with 5 Button Presses", "tap Countdown Sound"])]),

    dict(id="camera_remembers", family="camera", band=1,
         instruction="Every time I reopen the camera it throws away what I had set up. Make it come "
                     "back on the same mode I left it on, with my exposure tweak and my live-photo "
                     "choice still in place.",
         trap="'Creative Controls' is the fourth switch on the page and was not asked for",
         mirrors="Apple Support: Preserve camera settings",
         steps=[("settings.gen_preserve_settings", ["tap Camera Mode", "tap Exposure Adjustment",
                                                    "tap Live Photo"])]),

    # ---------------------------------------------------------------- band 2
    dict(id="tripsnap_quiet_notifications", family="notifications", band=2,
         instruction="TripSnap pings me constantly. I still want its notifications waiting for me in "
                     "the list when I look — it just shouldn't make a sound or stick a red dot on "
                     "the app icon.",
         trap="switching notifications off entirely is the obvious move and loses the ones they want to keep",
         mirrors="Apple Support: Change notification settings for an app",
         steps=[("settings.notif_app_TripSnap", ["tap Sounds", "tap Badges"])]),

    dict(id="keyboard_stop_meddling", family="keyboard", band=2,
         instruction="The keyboard keeps outsmarting me: it curls my quote marks, it pops a big "
                     "letter up over my thumb every time I press a key, and I keep knocking it into "
                     "capitals mid-sentence. Stop all three. Leave the spell-checking and the "
                     "corrections alone — those I want.",
         trap="Auto-Correction and Check Spelling are the switches people reach for and must stay on",
         mirrors="Apple Support: Keyboard settings",
         steps=[("settings.stub_keyboard", ["tap Smart Punctuation", "tap Character Preview",
                                            "tap Enable Caps Lock"])]),

    dict(id="assistive_touch_setup", family="accessibility", band=2,
         instruction="My hands shake and I can't press the physical buttons reliably any more. Put "
                     "the on-screen button on the display that can do those presses for me, and make "
                     "it ask me to confirm before it acts. Leave the pointer-key stuff alone — I "
                     "don't use a mouse.",
         trap="'Mouse Keys' is the third switch on the page and is explicitly not wanted",
         mirrors="Apple Support: AssistiveTouch",
         steps=[("settings.acc_assistive", ["tap AssistiveTouch", "tap Confirm with AssistiveTouch"])]),

    dict(id="focus_status_selective", family="focus", band=2,
         instruction="When I'm asleep or at work, people messaging me shouldn't be told my "
                     "notifications are silenced — but I'm happy for them to see it when I'm on Do "
                     "Not Disturb or my personal focus.",
         trap="the master 'Share Focus Status' switch turns it off for all four and over-shoots",
         mirrors="Apple Support: Share your Focus status",
         steps=[("settings.gen_focus_status", ["tap Sleep", "tap Work"])]),

    # ---------------------------------------------------------------- band 3
    dict(id="quickpay_permission_audit", family="privacy", band=3,
         instruction="QuickPay is for paying people. It shouldn't be able to listen through my "
                     "microphone, look through my camera, or read my address book. Take all three "
                     "away from it and leave every other app exactly as it is.",
         trap="the three permissions live on three different pages; the app's own page only shows two of them",
         mirrors="Apple Support: Control app access to microphone, camera and contacts",
         steps=[("settings.priv_mic", ["tap QuickPay"]),
                ("settings.priv_cam", ["tap QuickPay"]),
                ("settings.app_QuickPay", ["tap Contacts"])]),

    dict(id="nightstand_dark", family="display", band=3,
         instruction="The phone sits on my nightstand and keeps lighting the room up. It shouldn't "
                     "wake when I roll over or pick it up, and it shouldn't flash notifications "
                     "across the room while it's docked.",
         trap="Night Mode is already on and looks like the answer; turning StandBy off entirely "
              "removes the clock they want",
         mirrors="Apple Support: StandBy; Raise to Wake",
         steps=[("settings.stub_standby", ["tap Motion to Wake", "tap Show Notifications"]),
                ("settings.display", ["tap Raise to Wake"])]),

    dict(id="hearing_aid_setup", family="accessibility", band=3,
         instruction="I've just paired my hearing aids. I want the phone's own beeps and alerts sent "
                     "through them as well as calls, and I want the audio tuned for my hearing on "
                     "both calls and anything I play.",
         trap="the tuning lives on a different page from the hearing-device switches",
         mirrors="Apple Support: MFi hearing devices; Headphone Accommodations",
         steps=[("settings.gen_mfi_hearing_devices", ["tap Play System Sounds"]),
                ("settings.acc_headphone", ["tap Headphone Accommodations", "tap Phone", "tap Media"])]),

    dict(id="deaf_user_calls", family="accessibility", band=3,
         instruction="I'm deaf and I need this phone to work without sound. Let me hold calls as "
                     "text conversations and answer every incoming call that way automatically, and "
                     "make the camera light flash when something comes in. I also want a warning "
                     "when a fire alarm goes off near me.",
         trap="'Flash on Silent' and 'Flash While Unlocked' are neighbours of the alert flash and were not asked for",
         mirrors="Apple Support: RTT calls; LED Flash for Alerts; Sound Recognition",
         steps=[("settings.gen_rtt", ["tap Software RTT", "tap Answer All Calls as RTT"]),
                ("settings.acc_audiovisual", ["tap LED Flash for Alerts"]),
                ("settings.sr_sounds", ["tap Fire"])]),

    dict(id="kid_content_limits", family="screen_time", band=3,
         instruction="My nephew uses this phone at the weekend. He shouldn't be able to reach the "
                     "web through the browser or the camera, and the assistant shouldn't search the "
                     "web for him or read out anything with swearing in it.",
         trap="turning the whole assistant off is heavier than asked and breaks the phone for the owner",
         mirrors="Apple Support: Content & Privacy Restrictions",
         steps=[("settings.gen_allowed_apps_features", ["tap Safari", "tap Camera"]),
                ("settings.gen_intelligence_siri", ["tap Web Search Content", "tap Explicit Language"])]),

    # ---------------------------------------------------------------- band 4
    dict(id="roaming_full_lockdown", family="cellular", band=4,
         instruction="I'm abroad for a fortnight and roaming is extortionate. Only the browser and "
                     "the maps may touch mobile data, nothing may quietly refresh itself in the "
                     "background except my calendar. Mobile data itself stays on — I need those "
                     "two apps working.",
         trap="switching Cellular Data off at the top is the one move that breaks what must keep working",
         mirrors="Apple Support: Cellular data per app; Background App Refresh",
         steps=[("settings.cellular", ["tap TripSnap", "tap QuickPay", "tap Weatherly", "tap Music"]),
                ("settings.bg_refresh", ["tap TripSnap", "tap QuickPay", "tap Weatherly",
                                         "tap Safari", "tap Maps", "tap Music"])]),

    dict(id="low_vision_full_setup", family="accessibility", band=4,
         instruction="My father is losing his sight. Make the text heavy and high-contrast, mark the "
                     "switches and the buttons so he can see what is tappable, cut the see-through "
                     "layers that wash everything out, give him the screen magnifier that follows "
                     "what he's typing, and stop the screen locking on him after half a minute — "
                     "five minutes is about right. Don't invert the colours.",
         trap="Smart Invert and Classic Invert are the tempting answers and are explicitly refused; "
              "'Never' for locking leaves the phone permanently open",
         mirrors="Apple Support: Display & Text Size; Zoom; Auto-Lock",
         steps=[("settings.acc_display", ["tap Bold Text", "tap Increase Contrast",
                                          "tap On/Off Labels", "tap Button Shapes",
                                          "tap Reduce Transparency"]),
                ("settings.acc_zoom", ["tap Zoom", "tap Follow Focus", "tap Smart Typing"]),
                ("settings.display", ["tap Auto-Lock", "tap 5 Minutes"])]),

    dict(id="handing_phone_over", family="privacy", band=4,
         instruction="I'm lending this phone to a colleague for the afternoon. Nothing of mine should "
                     "be reachable from the locked screen — no notifications, no replying to my "
                     "messages, no paying with my cards, no calling people back, and no widgets. It "
                     "should also demand the passcode the moment the screen goes dark. Face ID must "
                     "still unlock it for me afterwards.",
         trap="turning iPhone Unlock off locks the owner out; Today View is the widget row people miss",
         mirrors="Apple Support: Allow access when locked; Require passcode",
         steps=[("settings.lockaccess", ["tap Notification Center", "tap Reply with Message",
                                         "tap Wallet", "tap Return Missed Calls",
                                         "tap Lock Screen Widgets", "tap Today View and Search"]),
                ("settings.stub_faceid", ["tap Require Passcode", "tap Immediately"])]),

    # ================================================================ batch 3 — value choices,
    # per-app notification variants, and deeper multi-area tasks
    # ---------------------------------------------------------------- band 1
    dict(id="airdrop_strangers_off", family="privacy", band=1, min_changes=1,
         instruction="People on the train keep trying to fling photos at my phone. Stop it accepting "
                     "anything from anyone at all — I'd rather turn it back on myself when I actually "
                     "want something.",
         trap="'Contacts Only' is the middle option and still accepts from people you know",
         mirrors="Apple Support: Use AirDrop",
         steps=[("settings.stub_airdrop", ["tap Receiving Off"])]),

    dict(id="weatherly_rough_location", family="privacy", band=1,
         instruction="Weatherly doesn't need to know exactly where I am — the neighbourhood is "
                     "plenty for a forecast — and I want it to check with me each time rather than "
                     "helping itself.",
         trap="'Never' breaks the forecast the owner still wants; the precise-location switch is a "
              "separate decision from the access mode",
         mirrors="Apple Support: Control location access and Precise Location",
         steps=[("settings.loc_app_Weatherly", ["tap Ask Next Time Or When I Share",
                                                "tap Precise Location"])]),

    dict(id="maps_navigates_in_pocket", family="privacy", band=1, min_changes=1,
         instruction="I keep missing my turn because the map stops following me the moment I put the "
                     "phone in my pocket. Let it keep using my location even when I'm not looking at it.",
         trap="'While Using the App' is the current setting and is exactly what breaks it",
         mirrors="Apple Support: Allow an app to use your location always",
         steps=[("settings.loc_app_Maps", ["tap Always"])]),

    dict(id="news_stops_shouting", family="notifications", band=1, min_changes=1,
         instruction="The news app interrupts me all day and I never want to hear from it again — "
                     "not a sound, not a banner, nothing.",
         trap="silencing only its sound leaves the banners coming",
         mirrors="Apple Support: Turn off notifications for an app",
         steps=[("settings.notif_app_News", ["tap Allow Notifications"])]),

    # ---------------------------------------------------------------- band 2
    dict(id="background_wifi_only", family="general", band=2, min_changes=1,
         instruction="Apps may carry on updating themselves in the background — but only when I'm on "
                     "wi-fi. Nothing should be doing that over my mobile connection. Don't stop them "
                     "refreshing altogether, and leave the individual apps' own settings as they are.",
         trap="'Off' is the neighbouring option and stops background refresh completely",
         mirrors="Apple Support: Background App Refresh",
         steps=[("settings.bg_global", ["tap Wi-Fi"])]),

    dict(id="mail_silent_but_visible", family="notifications", band=2,
         instruction="Mail should stop making noise and stop putting a red count on the icon — but I "
                     "still want the messages themselves arriving in my notification list.",
         trap="turning notifications off entirely removes what they want kept",
         mirrors="Apple Support: Notification settings per app",
         steps=[("settings.notif_app_Mail", ["tap Sounds", "tap Badges"])]),

    dict(id="volume_safety_for_kid", family="sounds", band=2,
         instruction="My son wears headphones for hours. Cap how loud the music can get and hold the "
                     "ceiling at the quietest setting the phone offers.",
         trap="the cap has its own switch as well as a level; setting one without the other does nothing",
         mirrors="Apple Support: Reduce loud sounds",
         steps=[("settings.sounds", ["tap Volume Limit", "tap Reduce Loud Audio",
                                     "tap Max Decibel Level", "tap 75 Decibels"])]),

    # ---------------------------------------------------------------- band 3
    dict(id="notification_triage", family="notifications", band=3,
         instruction="Only actual people should be allowed to interrupt me. Mail can keep arriving "
                     "but silently and without a count on the icon, and the news and the podcasts "
                     "shouldn't notify me at all any more.",
         trap="the three apps have separate notification pages; silencing mail entirely loses what is wanted",
         mirrors="Apple Support: Notification settings per app",
         steps=[("settings.notif_app_Mail", ["tap Sounds", "tap Badges"]),
                ("settings.notif_app_News", ["tap Allow Notifications"]),
                ("settings.notif_app_Podcasts", ["tap Allow Notifications"])]),

    dict(id="location_audit", family="privacy", band=3,
         instruction="Sort out which apps can see where I am. The travel diary should never get my "
                     "location again. The weather app should ask me each time and only get the "
                     "rough area. The map should be allowed to follow me even when I'm not in it, "
                     "because I use it for directions.",
         trap="each app has its own page and its own four options; 'Never' for the map breaks navigation",
         mirrors="Apple Support: Control app location access",
         steps=[("settings.loc_app_TripSnap", ["tap Never"]),
                ("settings.loc_app_Weatherly", ["tap Ask Next Time Or When I Share",
                                                "tap Precise Location"]),
                ("settings.loc_app_Maps", ["tap Always"])]),

    dict(id="standby_bedside_dark", family="display", band=3,
         instruction="When the phone is docked on my nightstand overnight, the clock should go dark "
                     "on its own instead of glowing all night, and it shouldn't light up because I "
                     "rolled over. I do still want the clock there when I look at it.",
         trap="'Never' for the display and turning StandBy off both remove the clock they want kept",
         mirrors="Apple Support: StandBy display options",
         steps=[("settings.stub_standby", ["tap Motion to Wake"]),
                ("settings.standby_display", ["tap Always On", "tap Turn Display Off",
                                              "tap After 20 Seconds"])]),

    dict(id="quickpay_location_and_notifications", family="privacy", band=3,
         instruction="The payments app is too nosy. It shouldn't know where I am at all, and it "
                     "shouldn't be allowed to put anything on my screen or make any noise. Leave "
                     "every other app's notifications exactly as they are.",
         trap="its location page and its notification page are in different parts of Settings",
         mirrors="Apple Support: App permissions; notification settings",
         steps=[("settings.loc_app_QuickPay", ["tap Never"]),
                ("settings.notif_app_QuickPay", ["tap Allow Notifications"])]),

    # ---------------------------------------------------------------- band 4
    dict(id="privacy_deep_clean", family="privacy", band=4,
         instruction="I want to stop being profiled. No app should even be able to ask to follow me "
                     "across other companies' apps, Apple should stop collecting how I use this "
                     "phone and stop keeping recordings of what I say to the assistant — but keep "
                     "sending the iCloud usage data, my work needs that. On top of that, the travel "
                     "diary should never get my location again and the weather app should have to "
                     "ask each time, with only the rough area.",
         trap="the iCloud analytics switch sits between the two that must go off and must stay on",
         mirrors="Apple Support: Tracking; Analytics & Improvements; Location Services",
         steps=[("settings.stub_tracking", ["tap Allow Apps to Request to Track"]),
                ("settings.analytics", ["tap Share iPhone Analytics", "tap Improve Siri & Dictation"]),
                ("settings.loc_app_TripSnap", ["tap Never"]),
                ("settings.loc_app_Weatherly", ["tap Ask Next Time Or When I Share",
                                                "tap Precise Location"])]),

    dict(id="mum_new_phone", family="accessibility", band=4,
         instruction="I'm setting my mother's new phone up. She can't read thin text and can't tell "
                     "which switches are on, the screen locks before she's finished reading it — "
                     "five minutes suits her — strangers keep sending her things she doesn't "
                     "understand so nothing should arrive from anyone nearby, and the news app "
                     "frightens her with alerts so it shouldn't notify her at all.",
         trap="'Contacts Only' still lets things through; 'Never' for the screen lock leaves the "
              "phone open on a bus",
         mirrors="Apple Support: Display & Text Size; Auto-Lock; AirDrop; notifications",
         steps=[("settings.acc_display", ["tap Bold Text", "tap Increase Contrast",
                                          "tap On/Off Labels"]),
                ("settings.display", ["tap Auto-Lock", "tap 5 Minutes"]),
                ("settings.stub_airdrop", ["tap Receiving Off"]),
                ("settings.notif_app_News", ["tap Allow Notifications"])]),

    # ================================================================ batch 4 — the screens and the
    # interaction types the benchmark uses that batches 1-3 never touched: the payments app (typing,
    # a one-time code), a confirmation dialog, a slider, iCloud, Siri, captions.
    dict(id="captions_on_the_train", family="accessibility", band=2,
         instruction="I watch things on the train with the sound off. Put the dialogue on screen, "
                     "and the other sounds too — doors, music, that sort of thing — written out as text.",
         trap="the captioning style page looks like the place to switch captions on and only changes how they look",
         mirrors="Apple Support: Subtitles & Captioning",
         steps=[("settings.acc_subtitles", ["tap Closed Captions + SDH", "tap Show Audio Transcriptions"])]),

    dict(id="bigger_text_slider", family="accessibility", band=2,
         instruction="The text is still too small for me even at the biggest normal setting. Unlock "
                     "the larger sizes and push the size up a few steps from where it is.",
         trap="the slider does nothing about the ceiling; the switch above it is what unlocks the bigger sizes",
         mirrors="Apple Support: Larger Accessibility Sizes",
         steps=[("raw", ["tap Accessibility", "tap Display & Text Size", "tap Larger Text",
                         "tap Larger Accessibility Sizes", "swipe right on Text Size",
                         "swipe right on Text Size", "swipe right on Text Size"])]),

    dict(id="siri_type_not_talk", family="siri", band=2,
         instruction="I'd rather type to the assistant than talk to it in public, and it shouldn't "
                     "answer anyone at all until the phone has been unlocked.",
         trap="turning Apple Intelligence off at the top is heavier than asked and kills the assistant entirely",
         mirrors="Apple Support: Type to Siri; Allow Siri When Locked",
         steps=[("raw", ["swipe up", "swipe up", "swipe up", "tap Siri", "tap Talk & Type to Siri",
                         "tap Type to Siri", "tap Allow Siri When Locked"])]),

    dict(id="icloud_selective_sync", family="icloud", band=3,
         instruction="I don't want my photos or my mail copied up to iCloud any more — they take up "
                     "the whole plan. My files and my passwords should carry on syncing exactly as "
                     "they do now.",
         trap="'Manage Account Storage' and 'See All' both look like the place to do this; the two "
              "switches that must stay on sit directly beneath the two that go off",
         mirrors="Apple Support: Choose what to store in iCloud",
         steps=[("raw", ["tap Chris Morgan", "tap iCloud", "tap Photos", "tap iCloud Mail"])]),

    dict(id="icloud_end_to_end", family="security", band=3,
         instruction="I want my iCloud locked so that not even Apple can read what's in it, and once "
                     "that's done my iCloud shouldn't be reachable through a web browser at all.",
         trap="the protection needs a confirmation before it is really on; web access is a separate "
              "switch that the protection does not turn off for you",
         mirrors="Apple Support: Advanced Data Protection for iCloud",
         steps=[("raw", ["tap Chris Morgan", "tap iCloud", "swipe up",
                         "tap Advanced Data Protection", "tap Turn On Advanced Data Protection",
                         "tap Turn On", "press back", "tap Access iCloud Data on the Web"])]),

    dict(id="stolen_device_guard", family="security", band=3, min_changes=1,
         instruction="If a pickpocket watches me type my passcode and then takes the phone, they "
                     "shouldn't be able to change my account or wipe it. Switch on the protection "
                     "that puts a delay on those when I'm somewhere unfamiliar.",
         trap="it lives at the bottom of the passcode page, under the lock-screen switches people stop at",
         mirrors="Apple Support: Stolen Device Protection",
         steps=[("raw", ["swipe up", "swipe up", "tap Face ID & Passcode", "swipe up",
                         "tap Stolen Device Protection", "tap Stolen Device Protection"])]),

    dict(id="pay_friend_no_fee", family="payments", band=3, start_app="bank",
         instruction="I owe Alex Kim a hundred and twenty for the concert tickets. Send it from my "
                     "QuickPay balance, and I'm not paying a fee for it — if it takes a few days to "
                     "land, fine. Put a note on it so he knows what it's for.",
         trap="the instant option is preselected and charges a fee; the card below the balance charges "
              "another one",
         mirrors="QuickPay: send money with a note",
         steps=[("raw", ["tap Send money", "tap Alex Kim", "tap the amount box", 'type "120"',
                         "press back", "tap the Note field", 'type "concert tickets"', "press back",
                         "tap 1-3 business days", "swipe up", "tap Review", "tap Confirm & send",
                         "tap the code field", "type OTP", "tap Verify & send"])]),

    dict(id="pay_landlord_today", family="payments", band=4, start_app="bank",
         instruction="Rent is late. Send the landlord nine hundred and fifty, label it October rent, "
                     "and it has to land today — I'll wear the fee. Take it off my Visa, not my "
                     "QuickPay balance; I need that balance for the deposit.",
         trap="the balance is preselected and is exactly what must not be used; the no-fee speed "
              "would miss the deadline",
         mirrors="QuickPay: choose a funding source and speed",
         steps=[("raw", ["tap Send money", "tap Landlord — Sunset Apts", "tap the amount box",
                         'type "950"', "press back", "tap the Note field", 'type "October rent"',
                         "press back", "tap Visa ···· 4821 · 3% fee", "swipe up", "tap Review",
                         "tap Confirm & send", "tap the code field", "type OTP", "tap Verify & send"])]),

    # ================================================================ batch 5
    # ---------------------------------------------------------------- band 1
    dict(id="background_sounds_focus", family="accessibility", band=1,
         instruction="Open-plan offices wreck my concentration. Give me the steady background noise "
                     "the phone can play, and keep it going underneath whatever else I'm listening to.",
         trap="the volume slider and the sound choice are not what switches it on",
         mirrors="Apple Support: Background Sounds",
         steps=[("settings.acc_bgsounds", ["tap Background Sounds", "tap Use When Media Is Playing"])]),

    dict(id="music_offline_only", family="apps", band=1,
         instruction="My music app eats my data allowance. Stop it streaming over the mobile network "
                     "and stop it updating itself in the background.",
         trap="the same two switches exist on the Cellular page; changing them twice undoes the work",
         mirrors="Apple Support: Cellular data per app",
         steps=[("settings.app_Music", ["tap Cellular Data", "tap Background App Refresh"])]),

    dict(id="home_screen_declutter", family="general", band=1,
         instruction="New apps I download shouldn't land on my home screen — keep them in the app "
                     "library — but I do want the little red counts on the icons so I know what's waiting.",
         trap="'Show in App Library' is already on and is not the setting that controls where new apps land",
         mirrors="Apple Support: Home Screen & App Library",
         steps=[("settings.stub_homescreen", ["tap App Library Only", "tap Show Notification Badges"])]),

    dict(id="wallet_requires_me", family="payments", band=1,
         instruction="I don't want this phone able to pay for anything unless I've unlocked it first "
                     "— no tapping through on a turnstile, no shortcut on the side button.",
         trap="the cards themselves stay on the phone; it is the two shortcuts that must go",
         mirrors="Apple Support: Express Transit; double-click to pay",
         steps=[("settings.wallet", ["tap Express Transit", "tap Double-Click Side Button"])]),

    dict(id="weatherly_trim", family="privacy", band=1,
         instruction="The weather app has my address book for no reason I can see, and it keeps "
                     "waking itself up when I'm not using it. Take both away.",
         trap="its camera and microphone are already off; the location page is a separate decision",
         mirrors="Apple Support: App permissions",
         steps=[("settings.app_Weatherly", ["tap Contacts", "tap Background App Refresh"])]),

    # ---------------------------------------------------------------- band 2
    dict(id="announce_only_headphones", family="siri", band=2, min_changes=1,
         instruction="When my headphones are in, I want the assistant to read my notifications out "
                     "to me. It should never do that out loud on the speaker, and it must always "
                     "check with me before sending a reply.",
         trap="'Reply Without Confirmation' is the switch underneath and is exactly what they refuse",
         mirrors="Apple Support: Announce Notifications",
         steps=[("settings.gen_announce_notifications", ["tap Announce Notifications"])]),

    dict(id="focus_status_nobody", family="focus", band=2, min_changes=1,
         instruction="Nobody messaging me should be told that I've got my notifications silenced — "
                     "not for any of my focus modes. Do it in one move rather than going through "
                     "them one at a time; the individual ones should stay as they are.",
         trap="switching the four modes off one by one looks equivalent and leaves the master on",
         mirrors="Apple Support: Share your Focus status",
         steps=[("settings.gen_focus_status", ["tap Share Focus Status"])]),

    dict(id="rtt_manual_send", family="accessibility", band=2,
         instruction="Turn on the text-based calling so I can type during a call — but I want to "
                     "finish my sentence before it goes, not have every letter sent as I type it. "
                     "Normal calls should still come through as calls.",
         trap="'Answer All Calls as RTT' is the third switch and turns every incoming call into text",
         mirrors="Apple Support: RTT calls",
         steps=[("settings.gen_rtt", ["tap Software RTT", "tap Send Immediately"])]),

    dict(id="safari_background_off", family="apps", band=2,
         instruction="The browser shouldn't be doing anything when I'm not looking at it — no "
                     "refreshing itself, no using my mobile data. Leave its pop-up blocking and "
                     "tracking protection exactly as they are.",
         trap="Block Pop-ups and Prevent Cross-Site Tracking sit right below and must not move",
         mirrors="Apple Support: Background App Refresh; cellular data per app",
         steps=[("settings.app_Safari", ["tap Background App Refresh", "tap Cellular Data"])]),

    dict(id="camera_no_qr", family="camera", band=2,
         instruction="I don't want the camera jumping at QR codes while I'm trying to take a photo, "
                     "and I'd rather it took its time and got the shot right than rushed it. Leave "
                     "the lens correction on.",
         trap="'Lens Correction' is the neighbouring switch and is explicitly kept",
         mirrors="Apple Support: Camera settings",
         steps=[("settings.stub_camera", ["tap Scan QR Codes", "tap Prioritize Faster Shooting"])]),

    dict(id="wallet_notifications_quiet", family="notifications", band=2,
         instruction="The wallet app should stop making a noise and stop putting a count on its "
                     "icon. I still want to see its notifications when I check.",
         trap="turning its notifications off entirely loses the receipts they want to see",
         mirrors="Apple Support: Notification settings per app",
         steps=[("settings.notif_app_Wallet", ["tap Sounds", "tap Badges"])]),

    # ---------------------------------------------------------------- band 3
    dict(id="browser_goes_quiet", family="privacy", band=3,
         instruction="The browser should stop knowing where I am, stop waking itself up in the "
                     "background, and stop notifying me altogether. Everything else about it stays "
                     "as it is.",
         trap="its location, its background refresh and its notifications live on three different pages",
         mirrors="Apple Support: App permissions and notifications",
         steps=[("settings.loc_app_Safari", ["tap Never"]),
                ("settings.app_Safari", ["tap Background App Refresh"]),
                ("settings.notif_app_Safari", ["tap Allow Notifications"])]),

    dict(id="music_data_discipline", family="apps", band=3,
         instruction="My music app is the reason I run out of data. It shouldn't stream over mobile, "
                     "it shouldn't refresh in the background, and it shouldn't make a sound when it "
                     "notifies me. Its notifications can still appear.",
         trap="its background refresh appears both on its own page and in the General list",
         mirrors="Apple Support: Cellular data; Background App Refresh; notifications",
         steps=[("settings.app_Music", ["tap Cellular Data"]),
                ("settings.bg_refresh", ["tap Music"]),
                ("settings.notif_app_Music", ["tap Sounds"])]),

    dict(id="voice_control_numbers_only", family="accessibility", band=3,
         instruction="I drive my phone by voice. Turn that on, and when it labels things on screen I "
                     "want the numbers only — not the names, not the grid over everything.",
         trap="the overlay choices are on a page two levels below the switch that turns it on",
         mirrors="Apple Support: Voice Control overlays",
         steps=[("settings.acc_voicecontrol", ["tap Voice Control"]),
                ("settings.gen_overlays", ["tap Show Names", "tap Show Grid"])]),

    dict(id="photos_private", family="privacy", band=3,
         instruction="My photo library should stop sharing albums with anyone, stop using mobile "
                     "data, and stop notifying me about memories and the rest of it.",
         trap="'Shared Library' is a row that looks like the sharing switch and is not it",
         mirrors="Apple Support: Shared Albums; app notifications",
         steps=[("settings.app_Photos", ["tap Shared Albums", "tap Background App Refresh"]),
                ("settings.notif_app_Photos", ["tap Allow Notifications"])]),

    dict(id="deaf_alerting_profile", family="accessibility", band=3,
         instruction="I can't hear the phone at all. Let me hold calls as typed text and have it "
                     "answer every call that way, flash the camera light at me even when the phone "
                     "isn't on silent, and warn me if a smoke alarm or a dog goes off near me.",
         trap="the flash settings and the sound warnings are in different corners of Accessibility",
         mirrors="Apple Support: RTT; LED Flash for Alerts; Sound Recognition",
         steps=[("settings.gen_rtt", ["tap Software RTT", "tap Answer All Calls as RTT"]),
                ("settings.acc_audiovisual", ["tap Flash on Silent"]),
                ("settings.sr_sounds", ["tap Fire", "tap Dog"])]),

    dict(id="commuter_maps", family="privacy", band=3,
         instruction="I want the map to follow me even when the phone is in my pocket and to be "
                     "allowed to interrupt me with time-critical directions. At the same time, "
                     "nothing on this phone should refresh itself over mobile data — wi-fi only.",
         trap="'Off' for background refresh stops it everywhere, which is more than was asked",
         mirrors="Apple Support: Location Always; Time-Sensitive notifications; Background App Refresh",
         steps=[("settings.loc_app_Maps", ["tap Always"]),
                ("settings.notif_app_Maps", ["tap Time-Sensitive Notifications"]),
                ("settings.bg_global", ["tap Wi-Fi"])]),

    # ---------------------------------------------------------------- band 4
    dict(id="payments_need_unlock", family="payments", band=4,
         instruction="Nobody should be able to spend money with this phone without unlocking it "
                     "first. No paying on a turnstile, no shortcut on the side button, no cards "
                     "reachable from the locked screen, and face recognition shouldn't authorise "
                     "payments either. Face recognition can still unlock the phone itself.",
         trap="turning iPhone Unlock off is the heavy-handed version and locks the owner out",
         mirrors="Apple Support: Wallet & Apple Pay; Allow access when locked; Face ID",
         steps=[("settings.wallet", ["tap Express Transit", "tap Double-Click Side Button"]),
                ("settings.lockaccess", ["tap Wallet"]),
                ("settings.stub_faceid", ["tap Wallet & Apple Pay"])]),

    dict(id="motor_impairment_setup", family="accessibility", band=4,
         instruction="My hands barely work any more. Put the on-screen button on the display that "
                     "can press the hardware buttons for me and make it confirm before acting, turn "
                     "on the scanning control so I can drive the phone with a switch, and stop the "
                     "trackpad gestures that I keep triggering by accident — the pinch and the "
                     "rotate ones.",
         trap="Switch Control's own sub-switches look like the ones to change and are not",
         mirrors="Apple Support: AssistiveTouch; Switch Control; Full Keyboard Access",
         steps=[("settings.acc_assistive", ["tap AssistiveTouch", "tap Confirm with AssistiveTouch"]),
                ("settings.acc_switch", ["tap Switch Control"]),
                ("settings.gen_hand_gestures", ["tap Pinch", "tap Rotate"])]),

    dict(id="loaner_phone_setup", family="privacy", band=4,
         instruction="I'm lending this to a contractor for a week. He shouldn't be able to reach the "
                     "web or the camera, the assistant shouldn't search the web for him or read out "
                     "anything with swearing in it, nothing should arrive from strangers standing "
                     "nearby, and the news app shouldn't bother him at all.",
         trap="'Contacts Only' still accepts from people in the address book",
         mirrors="Apple Support: Content & Privacy Restrictions; AirDrop; notifications",
         steps=[("settings.gen_allowed_apps_features", ["tap Safari", "tap Camera"]),
                ("settings.gen_intelligence_siri", ["tap Web Search Content", "tap Explicit Language"]),
                ("settings.stub_airdrop", ["tap Receiving Off"]),
                ("settings.notif_app_News", ["tap Allow Notifications"])]),
]
