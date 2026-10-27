"""Hand-written tasks for RL, compiled by tools/build_rl_tasks.py — do not edit by hand.

The wording, the situation and the choice of controls are hand-written (tools/rl_task_specs.py);
the plans and the goal states below are what the engine produced and verified by replay. `band` is
the complexity band: 1 = one page, a couple of controls; 2 = one page with a leave-alone
constraint; 3 = benchmark parity (several pages, a trap); 4 = above benchmark.
"""
from .tasks import g_paths, task


task(id='airdrop_strangers_off', family='privacy', band=1,
     instruction="People on the train keep trying to fling photos at my phone. Stop it accepting anything from anyone at all — I'd rather turn it back on myself when I actually want something.",
     start_app='settings', seed=0,
     mirrors='Apple Support: Use AirDrop',
     trap="'Contacts Only' is the middle option and still accepts from people you know",
     plan=['tap General', 'tap AirDrop', 'tap Receiving Off'],
     goal=g_paths({'settings.spec.airdrop': 'Receiving Off'}))

task(id='background_sounds_focus', family='accessibility', band=1,
     instruction="Open-plan offices wreck my concentration. Give me the steady background noise the phone can play, and keep it going underneath whatever else I'm listening to.",
     start_app='settings', seed=0,
     mirrors='Apple Support: Background Sounds',
     trap='the volume slider and the sound choice are not what switches it on',
     plan=['tap Accessibility', 'swipe up', 'swipe up', 'swipe up', 'tap Audio & Visual', 'tap Background Sounds', 'tap Background Sounds', 'tap Use When Media Is Playing'],
     goal=g_paths({'settings.spec.accessibility.use_when_media_is_playing': True, 'settings.spec.accessibility.bg_sounds': True}))

task(id='camera_framing_guides', family='camera', band=1,
     instruction='My photos come out crooked and my selfies come out flipped the wrong way round from what I saw while taking them. Put the lines across the viewfinder that split it into thirds so I can line a shot up, and make the front camera save selfies the way they looked on screen. Leave the rest of the camera alone.',
     start_app='settings', seed=0,
     mirrors='Apple Support: Take photos with the grid; Mirror front camera',
     trap="'Level' is already on and looks like the framing aid; 'View Outside the Frame' is not it",
     plan=['tap Camera', 'tap Grid', 'tap Mirror Front Camera'],
     goal=g_paths({'settings.spec.camera.grid': True, 'settings.spec.camera.mirror': True}))

task(id='camera_remembers', family='camera', band=1,
     instruction='Every time I reopen the camera it throws away what I had set up. Make it come back on the same mode I left it on, with my exposure tweak and my live-photo choice still in place.',
     start_app='settings', seed=0,
     mirrors='Apple Support: Preserve camera settings',
     trap="'Creative Controls' is the fourth switch on the page and was not asked for",
     plan=['tap Camera', 'tap Preserve Settings', 'tap Camera Mode', 'tap Exposure Adjustment', 'tap Live Photo'],
     goal=g_paths({'settings.spec.gen.preserve_settings.camera_mode': True, 'settings.spec.gen.preserve_settings.exposure_adjustment': True, 'settings.spec.gen.preserve_settings.live_photo': True}))

task(id='home_screen_declutter', family='general', band=1,
     instruction="New apps I download shouldn't land on my home screen — keep them in the app library — but I do want the little red counts on the icons so I know what's waiting.",
     start_app='settings', seed=0,
     mirrors='Apple Support: Home Screen & App Library',
     trap="'Show in App Library' is already on and is not the setting that controls where new apps land",
     plan=['swipe up', 'tap Home Screen & App Library', 'tap App Library Only', 'tap Show Notification Badges'],
     goal=g_paths({'settings.spec.homescreen.newapps': 'App Library Only', 'settings.spec.homescreen.badges_library': True}))

task(id='mail_thread_tidy', family='mail', band=1,
     instruction="When I open my mail I can't tell who else was on a message, and replies get bundled up so I lose track of them. Show me who a message went to besides me, and stop grouping the replies together.",
     start_app='settings', seed=0,
     mirrors='Apple Support: Mail settings — To/Cc labels, threading',
     trap="'Ask Before Deleting' is the other switch on that page and has nothing to do with it",
     plan=['swipe up', 'swipe up', 'swipe up', 'tap Apps', 'tap Mail', 'tap Show To/Cc Labels', 'swipe up', 'tap Organize by Thread'],
     goal=g_paths({'settings.spec.appx.mail_to_cc': True, 'settings.spec.appx.mail_thread': False}))

task(id='maps_navigates_in_pocket', family='privacy', band=1,
     instruction="I keep missing my turn because the map stops following me the moment I put the phone in my pocket. Let it keep using my location even when I'm not looking at it.",
     start_app='settings', seed=0,
     mirrors='Apple Support: Allow an app to use your location always',
     trap="'While Using the App' is the current setting and is exactly what breaks it",
     plan=['swipe up', 'swipe up', 'tap Privacy & Security', 'tap Location Services', 'tap Maps', 'tap Always'],
     goal=g_paths({'settings.apps.Maps.loc_mode': 'Always'}))

task(id='meeting_manners', family='sounds', band=1,
     instruction="In meetings I want the phone to buzz even when it's on silent, and I want it to stop clicking at everyone every time I type.",
     start_app='settings', seed=0,
     mirrors='Apple Support: Sounds & Haptics',
     trap="'Vibrate on Ring' is already on and is the other, wrong, vibration switch",
     plan=['swipe up', 'swipe up', 'tap Sounds & Haptics', 'tap Vibrate on Silent', 'tap Keyboard Clicks'],
     goal=g_paths({'settings.sound.vibrate_silent': True, 'settings.sound.keyboard_clicks': False}))

task(id='motion_calm_basic', family='accessibility', band=1,
     instruction="Sliding animations and anything that strobes set off my migraines. Cut the movement in the interface right down, dim the flashing, and hold the screen refresh at the lower rate. I still want message effects and video previews to play by themselves — don't touch those.",
     start_app='settings', seed=0,
     mirrors='Apple Support: Reduce screen motion',
     trap="the auto-play switches sit in the same list and look like 'motion' too",
     plan=['tap Accessibility', 'tap Motion', 'tap Reduce Motion', 'tap Dim Flashing Lights', 'tap Limit Frame Rate'],
     goal=g_paths({'settings.spec.accessibility.reduce_motion': True, 'settings.spec.accessibility.limit_frame_rate': True, 'settings.spec.accessibility.dim_flashing': True}))

task(id='music_offline_only', family='apps', band=1,
     instruction='My music app eats my data allowance. Stop it streaming over the mobile network and stop it updating itself in the background.',
     start_app='settings', seed=0,
     mirrors='Apple Support: Cellular data per app',
     trap='the same two switches exist on the Cellular page; changing them twice undoes the work',
     plan=['swipe up', 'swipe up', 'swipe up', 'tap Apps', 'tap Music', 'tap Cellular Data', 'tap Background App Refresh'],
     goal=g_paths({'settings.apps.Music.cellular': False, 'settings.apps.Music.bg': False}))

task(id='news_stops_shouting', family='notifications', band=1,
     instruction='The news app interrupts me all day and I never want to hear from it again — not a sound, not a banner, nothing.',
     start_app='settings', seed=0,
     mirrors='Apple Support: Turn off notifications for an app',
     trap='silencing only its sound leaves the banners coming',
     plan=['swipe up', 'swipe up', 'swipe up', 'tap Apps', 'swipe up', 'tap News', 'tap Notifications', 'tap Allow Notifications'],
     goal=g_paths({'settings.apps.News.notif.allow': False}))

task(id='readability_basic', family='accessibility', band=1,
     instruction="My eyes are getting worse. Make the text heavier, push the contrast up, and put the little on/off markers inside the switches so I can tell at a glance which way each one is set. Don't invert the colours — I can't stand how that looks.",
     start_app='settings', seed=0,
     mirrors='Apple Support: Display & Text Size accommodations',
     trap="Smart Invert and Classic Invert are the tempting 'contrast' answers and break the request",
     plan=['tap Accessibility', 'tap Display & Text Size', 'tap Bold Text', 'tap Increase Contrast', 'tap On/Off Labels'],
     goal=g_paths({'settings.spec.accessibility.bold': True, 'settings.spec.accessibility.increase_contrast': True, 'settings.spec.accessibility.on_off_labels': True}))

task(id='sos_discreet', family='safety', band=1,
     instruction="If I'm ever in trouble I want to call the emergency services by pressing the side button five times — and I don't want the phone announcing it with a countdown noise while I'm doing it.",
     start_app='settings', seed=0,
     mirrors='Apple Support: Emergency SOS',
     trap="'Call with Hold and Release' is already on and is the other way of triggering the call",
     plan=['swipe up', 'swipe up', 'tap Emergency SOS', 'tap Call with 5 Button Presses', 'tap Countdown Sound'],
     goal=g_paths({'settings.spec.sos.call5': True, 'settings.spec.sos.countdown': False}))

task(id='tripsnap_permission_trim', family='privacy', band=1,
     instruction='TripSnap is a travel diary — it has no business with my camera or my address book. Cut both of those off. It may keep everything else it has.',
     start_app='settings', seed=0,
     mirrors='Apple Support: Control app access to camera and contacts',
     trap="the app's microphone switch is already off; turning off its data or refresh is not asked for",
     plan=['swipe up', 'swipe up', 'swipe up', 'tap Apps', 'swipe up', 'swipe up', 'tap TripSnap', 'tap Camera', 'tap Contacts'],
     goal=g_paths({'settings.apps.TripSnap.perms.Camera': False, 'settings.apps.TripSnap.perms.Contacts': False}))

task(id='wallet_requires_me', family='payments', band=1,
     instruction="I don't want this phone able to pay for anything unless I've unlocked it first — no tapping through on a turnstile, no shortcut on the side button.",
     start_app='settings', seed=0,
     mirrors='Apple Support: Express Transit; double-click to pay',
     trap='the cards themselves stay on the phone; it is the two shortcuts that must go',
     plan=['swipe up', 'swipe up', 'swipe up', 'tap Wallet & Apple Pay', 'tap Express Transit', 'tap Double-Click Side Button'],
     goal=g_paths({'settings.spec.wallet.double_click': False, 'settings.spec.wallet.express': False}))

task(id='weatherly_rough_location', family='privacy', band=1,
     instruction="Weatherly doesn't need to know exactly where I am — the neighbourhood is plenty for a forecast — and I want it to check with me each time rather than helping itself.",
     start_app='settings', seed=0,
     mirrors='Apple Support: Control location access and Precise Location',
     trap="'Never' breaks the forecast the owner still wants; the precise-location switch is a separate decision from the access mode",
     plan=['swipe up', 'swipe up', 'tap Privacy & Security', 'tap Location Services', 'tap Weatherly', 'tap Ask Next Time Or When I Share', 'tap Precise Location'],
     goal=g_paths({'settings.apps.Weatherly.loc_mode': 'Ask Next Time Or When I Share', 'settings.apps.Weatherly.precise': False}))

task(id='weatherly_trim', family='privacy', band=1,
     instruction="The weather app has my address book for no reason I can see, and it keeps waking itself up when I'm not using it. Take both away.",
     start_app='settings', seed=0,
     mirrors='Apple Support: App permissions',
     trap='its camera and microphone are already off; the location page is a separate decision',
     plan=['swipe up', 'swipe up', 'swipe up', 'tap Apps', 'swipe up', 'swipe up', 'swipe up', 'tap Weatherly', 'tap Contacts', 'tap Background App Refresh'],
     goal=g_paths({'settings.apps.Weatherly.perms.Contacts': False, 'settings.apps.Weatherly.bg': False}))

task(id='announce_only_headphones', family='siri', band=2,
     instruction='When my headphones are in, I want the assistant to read my notifications out to me. It should never do that out loud on the speaker, and it must always check with me before sending a reply.',
     start_app='settings', seed=0,
     mirrors='Apple Support: Announce Notifications',
     trap="'Reply Without Confirmation' is the switch underneath and is exactly what they refuse",
     plan=['swipe up', 'tap Siri', 'tap Announce Notifications', 'tap Announce Notifications'],
     goal=g_paths({'settings.spec.gen.announce_notifications.announce_notifications': True}))

task(id='assistive_touch_setup', family='accessibility', band=2,
     instruction="My hands shake and I can't press the physical buttons reliably any more. Put the on-screen button on the display that can do those presses for me, and make it ask me to confirm before it acts. Leave the pointer-key stuff alone — I don't use a mouse.",
     start_app='settings', seed=0,
     mirrors='Apple Support: AssistiveTouch',
     trap="'Mouse Keys' is the third switch on the page and is explicitly not wanted",
     plan=['tap Accessibility', 'tap Touch', 'tap AssistiveTouch', 'tap AssistiveTouch', 'tap Confirm with AssistiveTouch'],
     goal=g_paths({'settings.spec.acc_at.on': True, 'settings.spec.acc_at.confirm': True}))

task(id='autoplay_stop', family='accessibility', band=2,
     instruction="Every time I open a message the effects fly around the screen, and videos and animated pictures start playing on their own. Stop all three of those from playing by themselves. Don't slow the whole interface down though — the general motion setting stays as it is.",
     start_app='settings', seed=0,
     mirrors='Apple Support: Auto-play settings under Motion',
     trap='Reduce Motion looks like the one-switch answer and is explicitly not wanted',
     plan=['tap Accessibility', 'tap Motion', 'tap Auto-Play Message Effects', 'tap Auto-Play Video Previews', 'tap Auto-Play Animated Images'],
     goal=g_paths({'settings.spec.accessibility.autoplay_messages': False, 'settings.spec.accessibility.autoplay_previews': False, 'settings.spec.accessibility.autoplay_images': False}))

task(id='background_wifi_only', family='general', band=2,
     instruction="Apps may carry on updating themselves in the background — but only when I'm on wi-fi. Nothing should be doing that over my mobile connection. Don't stop them refreshing altogether, and leave the individual apps' own settings as they are.",
     start_app='settings', seed=0,
     mirrors='Apple Support: Background App Refresh',
     trap="'Off' is the neighbouring option and stops background refresh completely",
     plan=['tap General', 'tap Background App Refresh', 'tap Background App Refresh', 'tap Wi-Fi'],
     goal=g_paths({'settings.bg_global': 'Wi-Fi'}))

task(id='bigger_text_slider', family='accessibility', band=2,
     instruction='The text is still too small for me even at the biggest normal setting. Unlock the larger sizes and push the size up a few steps from where it is.',
     start_app='settings', seed=0,
     mirrors='Apple Support: Larger Accessibility Sizes',
     trap='the slider does nothing about the ceiling; the switch above it is what unlocks the bigger sizes',
     plan=['tap Accessibility', 'tap Display & Text Size', 'tap Larger Text', 'tap Larger Accessibility Sizes', 'swipe right on Text Size', 'swipe right on Text Size', 'swipe right on Text Size'],
     goal=g_paths({'settings.spec.accessibility.larger': True, 'settings.spec.accessibility.text_size': 100}))

task(id='camera_no_qr', family='camera', band=2,
     instruction="I don't want the camera jumping at QR codes while I'm trying to take a photo, and I'd rather it took its time and got the shot right than rushed it. Leave the lens correction on.",
     start_app='settings', seed=0,
     mirrors='Apple Support: Camera settings',
     trap="'Lens Correction' is the neighbouring switch and is explicitly kept",
     plan=['tap Camera', 'swipe up', 'tap Scan QR Codes', 'tap Prioritize Faster Shooting'],
     goal=g_paths({'settings.spec.camera.scan_qr': False, 'settings.spec.camera.prioritize_shot': False}))

task(id='captions_on_the_train', family='accessibility', band=2,
     instruction='I watch things on the train with the sound off. Put the dialogue on screen, and the other sounds too — doors, music, that sort of thing — written out as text.',
     start_app='settings', seed=0,
     mirrors='Apple Support: Subtitles & Captioning',
     trap='the captioning style page looks like the place to switch captions on and only changes how they look',
     plan=['tap Accessibility', 'swipe up', 'swipe up', 'tap Subtitles & Captioning', 'tap Closed Captions + SDH', 'tap Show Audio Transcriptions'],
     goal=g_paths({'settings.spec.acc_subs.closed_captions': True, 'settings.spec.acc_subs.show_audio_transcriptions': True}))

task(id='child_allowed_apps', family='screen_time', band=2,
     instruction="I'm handing this phone to my nephew for the afternoon. He shouldn't be able to get on the web, take photos, or fling files to strangers standing nearby. He does need to be able to reach me, so leave email and video calling working.",
     start_app='settings', seed=0,
     mirrors='Apple Support: Content & Privacy Restrictions — allowed apps',
     trap='turning off Mail or FaceTime breaks the one thing he must keep',
     plan=['swipe up', 'swipe up', 'tap Screen Time', 'tap Content & Privacy Restrictions', 'tap Allowed Apps & Features', 'tap Safari', 'tap Camera', 'tap AirDrop'],
     goal=g_paths({'settings.spec.gen.allowed_apps_features.airdrop': False, 'settings.spec.gen.allowed_apps_features.camera': False, 'settings.spec.gen.allowed_apps_features.safari': False}))

task(id='focus_status_nobody', family='focus', band=2,
     instruction="Nobody messaging me should be told that I've got my notifications silenced — not for any of my focus modes. Do it in one move rather than going through them one at a time; the individual ones should stay as they are.",
     start_app='settings', seed=0,
     mirrors='Apple Support: Share your Focus status',
     trap='switching the four modes off one by one looks equivalent and leaves the master on',
     plan=['swipe up', 'swipe up', 'tap Focus', 'tap Focus Status', 'tap Share Focus Status'],
     goal=g_paths({'settings.spec.gen.focus_status.share_focus_status': False}))

task(id='focus_status_selective', family='focus', band=2,
     instruction="When I'm asleep or at work, people messaging me shouldn't be told my notifications are silenced — but I'm happy for them to see it when I'm on Do Not Disturb or my personal focus.",
     start_app='settings', seed=0,
     mirrors='Apple Support: Share your Focus status',
     trap="the master 'Share Focus Status' switch turns it off for all four and over-shoots",
     plan=['swipe up', 'swipe up', 'tap Focus', 'tap Focus Status', 'tap Sleep', 'tap Work'],
     goal=g_paths({'settings.spec.gen.focus_status.sleep': False, 'settings.spec.gen.focus_status.work': False}))

task(id='keyboard_stop_meddling', family='keyboard', band=2,
     instruction='The keyboard keeps outsmarting me: it curls my quote marks, it pops a big letter up over my thumb every time I press a key, and I keep knocking it into capitals mid-sentence. Stop all three. Leave the spell-checking and the corrections alone — those I want.',
     start_app='settings', seed=0,
     mirrors='Apple Support: Keyboard settings',
     trap='Auto-Correction and Check Spelling are the switches people reach for and must stay on',
     plan=['tap General', 'tap Keyboard', 'tap Smart Punctuation', 'tap Character Preview', 'tap Enable Caps Lock'],
     goal=g_paths({'settings.spec.keyboard.caps_lock': False, 'settings.spec.keyboard.smart_punct': False, 'settings.spec.keyboard.character_preview': False}))

task(id='lockscreen_private', family='privacy', band=2,
     instruction="Anyone who picks my phone up can read my notifications and search through my stuff without unlocking it. Shut both of those off at the lock screen. The quick-settings panel has to keep working there — that's my flashlight.",
     start_app='settings', seed=0,
     mirrors='Apple Support: Allow access when locked',
     trap='Control Center is the switch people grab first and it must stay on',
     plan=['swipe up', 'swipe up', 'tap Face ID & Passcode', 'tap Allow Access When Locked', 'tap Notification Center', 'tap Today View and Search'],
     goal=g_paths({'settings.spec.lockaccess.today': False, 'settings.spec.lockaccess.notif_center': False}))

task(id='mail_silent_but_visible', family='notifications', band=2,
     instruction='Mail should stop making noise and stop putting a red count on the icon — but I still want the messages themselves arriving in my notification list.',
     start_app='settings', seed=0,
     mirrors='Apple Support: Notification settings per app',
     trap='turning notifications off entirely removes what they want kept',
     plan=['swipe up', 'tap Notifications', 'swipe up', 'tap Mail', 'tap Sounds', 'tap Badges'],
     goal=g_paths({'settings.apps.Mail.notif.sounds': False, 'settings.apps.Mail.notif.badges': False}))

task(id='rtt_manual_send', family='accessibility', band=2,
     instruction='Turn on the text-based calling so I can type during a call — but I want to finish my sentence before it goes, not have every letter sent as I type it. Normal calls should still come through as calls.',
     start_app='settings', seed=0,
     mirrors='Apple Support: RTT calls',
     trap="'Answer All Calls as RTT' is the third switch and turns every incoming call into text",
     plan=['tap Accessibility', 'swipe up', 'swipe up', 'tap RTT', 'tap Software RTT', 'tap Send Immediately'],
     goal=g_paths({'settings.spec.gen.rtt.send_immediately': False, 'settings.spec.gen.rtt.software_rtt': True}))

task(id='safari_background_off', family='apps', band=2,
     instruction="The browser shouldn't be doing anything when I'm not looking at it — no refreshing itself, no using my mobile data. Leave its pop-up blocking and tracking protection exactly as they are.",
     start_app='settings', seed=0,
     mirrors='Apple Support: Background App Refresh; cellular data per app',
     trap='Block Pop-ups and Prevent Cross-Site Tracking sit right below and must not move',
     plan=['swipe up', 'swipe up', 'swipe up', 'tap Apps', 'swipe up', 'swipe up', 'tap Safari', 'tap Background App Refresh', 'tap Cellular Data'],
     goal=g_paths({'settings.apps.Safari.cellular': False, 'settings.apps.Safari.bg': False}))

task(id='siri_type_not_talk', family='siri', band=2,
     instruction="I'd rather type to the assistant than talk to it in public, and it shouldn't answer anyone at all until the phone has been unlocked.",
     start_app='settings', seed=0,
     mirrors='Apple Support: Type to Siri; Allow Siri When Locked',
     trap='turning Apple Intelligence off at the top is heavier than asked and kills the assistant entirely',
     plan=['swipe up', 'swipe up', 'swipe up', 'tap Siri', 'tap Talk & Type to Siri', 'tap Type to Siri', 'tap Allow Siri When Locked'],
     goal=g_paths({'settings.spec.siri.type_to_siri': True, 'settings.spec.siri.when_locked': False}))

task(id='tripsnap_quiet_notifications', family='notifications', band=2,
     instruction="TripSnap pings me constantly. I still want its notifications waiting for me in the list when I look — it just shouldn't make a sound or stick a red dot on the app icon.",
     start_app='settings', seed=0,
     mirrors='Apple Support: Change notification settings for an app',
     trap='switching notifications off entirely is the obvious move and loses the ones they want to keep',
     plan=['swipe up', 'tap Notifications', 'tap TripSnap', 'tap Sounds', 'tap Badges'],
     goal=g_paths({'settings.apps.TripSnap.notif.sounds': False, 'settings.apps.TripSnap.notif.badges': False}))

task(id='volume_safety_for_kid', family='sounds', band=2,
     instruction='My son wears headphones for hours. Cap how loud the music can get and hold the ceiling at the quietest setting the phone offers.',
     start_app='settings', seed=0,
     mirrors='Apple Support: Reduce loud sounds',
     trap='the cap has its own switch as well as a level; setting one without the other does nothing',
     plan=['swipe up', 'swipe up', 'tap Sounds & Haptics', 'tap Volume Limit', 'tap Reduce Loud Audio', 'tap Max Decibel Level', 'tap 75 Decibels'],
     goal=g_paths({'settings.spec.gen.volume_limit.max_decibel_level': '75 Decibels', 'settings.spec.gen.volume_limit.reduce_loud_audio': True}))

task(id='wallet_notifications_quiet', family='notifications', band=2,
     instruction='The wallet app should stop making a noise and stop putting a count on its icon. I still want to see its notifications when I check.',
     start_app='settings', seed=0,
     mirrors='Apple Support: Notification settings per app',
     trap='turning its notifications off entirely loses the receipts they want to see',
     plan=['swipe up', 'swipe up', 'swipe up', 'tap Apps', 'swipe up', 'swipe up', 'tap Wallet', 'tap Notifications', 'tap Sounds', 'tap Badges'],
     goal=g_paths({'settings.apps.Wallet.notif.sounds': False, 'settings.apps.Wallet.notif.badges': False}))

task(id='browser_goes_quiet', family='privacy', band=3,
     instruction='The browser should stop knowing where I am, stop waking itself up in the background, and stop notifying me altogether. Everything else about it stays as it is.',
     start_app='settings', seed=0,
     mirrors='Apple Support: App permissions and notifications',
     trap='its location, its background refresh and its notifications live on three different pages',
     plan=['swipe up', 'swipe up', 'tap Privacy & Security', 'tap Location Services', 'tap Safari', 'tap Never', 'press back', 'press back', 'press back', 'swipe up', 'tap Apps', 'swipe up', 'swipe up', 'tap Safari', 'tap Background App Refresh', 'press back', 'press back', 'tap Notifications', 'tap Safari', 'tap Allow Notifications'],
     goal=g_paths({'settings.apps.Safari.loc_mode': 'Never', 'settings.apps.Safari.bg': False, 'settings.apps.Safari.notif.allow': False}))

task(id='commuter_maps', family='privacy', band=3,
     instruction='I want the map to follow me even when the phone is in my pocket and to be allowed to interrupt me with time-critical directions. At the same time, nothing on this phone should refresh itself over mobile data — wi-fi only.',
     start_app='settings', seed=0,
     mirrors='Apple Support: Location Always; Time-Sensitive notifications; Background App Refresh',
     trap="'Off' for background refresh stops it everywhere, which is more than was asked",
     plan=['swipe up', 'swipe up', 'tap Privacy & Security', 'tap Location Services', 'tap Maps', 'tap Always', 'press back', 'press back', 'press back', 'tap Notifications', 'tap Maps', 'tap Time-Sensitive Notifications', 'press back', 'press back', 'swipe up', 'swipe up', 'swipe up', 'swipe up', 'swipe up', 'swipe up', 'swipe up', 'swipe up', 'swipe down', 'tap General', 'tap Background App Refresh', 'tap Background App Refresh', 'tap Wi-Fi'],
     goal=g_paths({'settings.bg_global': 'Wi-Fi', 'settings.apps.Maps.loc_mode': 'Always', 'settings.apps.Maps.notif.time_sensitive': True}))

task(id='deaf_alerting_profile', family='accessibility', band=3,
     instruction="I can't hear the phone at all. Let me hold calls as typed text and have it answer every call that way, flash the camera light at me even when the phone isn't on silent, and warn me if a smoke alarm or a dog goes off near me.",
     start_app='settings', seed=0,
     mirrors='Apple Support: RTT; LED Flash for Alerts; Sound Recognition',
     trap='the flash settings and the sound warnings are in different corners of Accessibility',
     plan=['tap Accessibility', 'swipe up', 'swipe up', 'tap RTT', 'tap Software RTT', 'tap Answer All Calls as RTT', 'press back', 'swipe up', 'tap Audio & Visual', 'tap Flash on Silent', 'press back', 'tap Sound Recognition', 'tap Sounds', 'tap Fire', 'tap Dog'],
     goal=g_paths({'settings.spec.accessibility.flash_on_silent': True, 'settings.spec.acc_sr.fire': True, 'settings.spec.acc_sr.dog': True, 'settings.spec.gen.rtt.answer_all_calls_as_rtt': True, 'settings.spec.gen.rtt.software_rtt': True}))

task(id='deaf_user_calls', family='accessibility', band=3,
     instruction="I'm deaf and I need this phone to work without sound. Let me hold calls as text conversations and answer every incoming call that way automatically, and make the camera light flash when something comes in. I also want a warning when a fire alarm goes off near me.",
     start_app='settings', seed=0,
     mirrors='Apple Support: RTT calls; LED Flash for Alerts; Sound Recognition',
     trap="'Flash on Silent' and 'Flash While Unlocked' are neighbours of the alert flash and were not asked for",
     plan=['tap Accessibility', 'swipe up', 'swipe up', 'tap RTT', 'tap Software RTT', 'tap Answer All Calls as RTT', 'press back', 'swipe up', 'tap Audio & Visual', 'tap LED Flash for Alerts', 'press back', 'tap Sound Recognition', 'tap Sounds', 'tap Fire'],
     goal=g_paths({'settings.spec.accessibility.led_flash': True, 'settings.spec.acc_sr.fire': True, 'settings.spec.gen.rtt.answer_all_calls_as_rtt': True, 'settings.spec.gen.rtt.software_rtt': True}))

task(id='hearing_aid_setup', family='accessibility', band=3,
     instruction="I've just paired my hearing aids. I want the phone's own beeps and alerts sent through them as well as calls, and I want the audio tuned for my hearing on both calls and anything I play.",
     start_app='settings', seed=0,
     mirrors='Apple Support: MFi hearing devices; Headphone Accommodations',
     trap='the tuning lives on a different page from the hearing-device switches',
     plan=['tap Accessibility', 'swipe up', 'swipe up', 'tap Hearing Devices', 'tap MFi Hearing Devices', 'tap Play System Sounds', 'press back', 'press back', 'swipe up', 'tap Audio & Visual', 'tap Headphone Accommodations', 'tap Headphone Accommodations', 'tap Phone', 'tap Media'],
     goal=g_paths({'settings.spec.accessibility.phone': True, 'settings.spec.accessibility.media': True, 'settings.spec.accessibility.headphone_acc': True, 'settings.spec.gen.mfi_hearing_devices.play_system_sounds': True}))

task(id='icloud_end_to_end', family='security', band=3,
     instruction="I want my iCloud locked so that not even Apple can read what's in it, and once that's done my iCloud shouldn't be reachable through a web browser at all.",
     start_app='settings', seed=0,
     mirrors='Apple Support: Advanced Data Protection for iCloud',
     trap='the protection needs a confirmation before it is really on; web access is a separate switch that the protection does not turn off for you',
     plan=['tap Chris Morgan', 'tap iCloud', 'swipe up', 'tap Advanced Data Protection', 'tap Turn On Advanced Data Protection', 'tap Turn On', 'press back', 'tap Access iCloud Data on the Web'],
     goal=g_paths({'settings.spec.icloud.web_access': False, 'settings.spec.icloud.adp': True}))

task(id='icloud_selective_sync', family='icloud', band=3,
     instruction="I don't want my photos or my mail copied up to iCloud any more — they take up the whole plan. My files and my passwords should carry on syncing exactly as they do now.",
     start_app='settings', seed=0,
     mirrors='Apple Support: Choose what to store in iCloud',
     trap="'Manage Account Storage' and 'See All' both look like the place to do this; the two switches that must stay on sit directly beneath the two that go off",
     plan=['tap Chris Morgan', 'tap iCloud', 'tap Photos', 'tap iCloud Mail'],
     goal=g_paths({'settings.spec.icloud.photos': False, 'settings.spec.icloud.mail': False}))

task(id='kid_content_limits', family='screen_time', band=3,
     instruction="My nephew uses this phone at the weekend. He shouldn't be able to reach the web through the browser or the camera, and the assistant shouldn't search the web for him or read out anything with swearing in it.",
     start_app='settings', seed=0,
     mirrors='Apple Support: Content & Privacy Restrictions',
     trap='turning the whole assistant off is heavier than asked and breaks the phone for the owner',
     plan=['swipe up', 'swipe up', 'tap Screen Time', 'tap Content & Privacy Restrictions', 'tap Allowed Apps & Features', 'tap Safari', 'tap Camera', 'press back', 'tap Intelligence & Siri', 'tap Web Search Content', 'tap Explicit Language'],
     goal=g_paths({'settings.spec.gen.allowed_apps_features.camera': False, 'settings.spec.gen.allowed_apps_features.safari': False, 'settings.spec.gen.intelligence_siri.explicit_language': False, 'settings.spec.gen.intelligence_siri.web_search_content': False}))

task(id='location_audit', family='privacy', band=3,
     instruction="Sort out which apps can see where I am. The travel diary should never get my location again. The weather app should ask me each time and only get the rough area. The map should be allowed to follow me even when I'm not in it, because I use it for directions.",
     start_app='settings', seed=0,
     mirrors='Apple Support: Control app location access',
     trap="each app has its own page and its own four options; 'Never' for the map breaks navigation",
     plan=['swipe up', 'swipe up', 'tap Privacy & Security', 'tap Location Services', 'tap TripSnap', 'tap Never', 'press back', 'tap Weatherly', 'tap Ask Next Time Or When I Share', 'tap Precise Location', 'press back', 'tap Maps', 'tap Always'],
     goal=g_paths({'settings.apps.TripSnap.loc_mode': 'Never', 'settings.apps.Weatherly.loc_mode': 'Ask Next Time Or When I Share', 'settings.apps.Weatherly.precise': False, 'settings.apps.Maps.loc_mode': 'Always'}))

task(id='motion_calm_full', family='accessibility', band=3,
     instruction="Since my concussion, movement on the screen makes me nauseous. Take the motion out of the interface, stop anything that plays or animates on its own, and dim whatever flashes. Leave the screen refresh rate alone — that one doesn't bother me.",
     start_app='settings', seed=0,
     mirrors='Apple Support: Reduce screen motion; auto-play settings',
     trap='Limit Frame Rate is in the same list and is explicitly not wanted',
     plan=['tap Accessibility', 'tap Motion', 'tap Reduce Motion', 'tap Auto-Play Message Effects', 'tap Auto-Play Video Previews', 'tap Auto-Play Animated Images', 'tap Dim Flashing Lights'],
     goal=g_paths({'settings.spec.accessibility.reduce_motion': True, 'settings.spec.accessibility.autoplay_messages': False, 'settings.spec.accessibility.autoplay_previews': False, 'settings.spec.accessibility.autoplay_images': False, 'settings.spec.accessibility.dim_flashing': True}))

task(id='music_data_discipline', family='apps', band=3,
     instruction="My music app is the reason I run out of data. It shouldn't stream over mobile, it shouldn't refresh in the background, and it shouldn't make a sound when it notifies me. Its notifications can still appear.",
     start_app='settings', seed=0,
     mirrors='Apple Support: Cellular data; Background App Refresh; notifications',
     trap='its background refresh appears both on its own page and in the General list',
     plan=['swipe up', 'swipe up', 'swipe up', 'tap Apps', 'tap Music', 'tap Cellular Data', 'press back', 'press back', 'swipe up', 'swipe up', 'swipe up', 'swipe up', 'swipe up', 'swipe up', 'swipe up', 'swipe up', 'swipe down', 'tap General', 'tap Background App Refresh', 'tap Music', 'press back', 'press back', 'tap Notifications', 'tap Music', 'tap Sounds'],
     goal=g_paths({'settings.apps.Music.cellular': False, 'settings.apps.Music.bg': False, 'settings.apps.Music.notif.sounds': False}))

task(id='nightstand_dark', family='display', band=3,
     instruction="The phone sits on my nightstand and keeps lighting the room up. It shouldn't wake when I roll over or pick it up, and it shouldn't flash notifications across the room while it's docked.",
     start_app='settings', seed=0,
     mirrors='Apple Support: StandBy; Raise to Wake',
     trap='Night Mode is already on and looks like the answer; turning StandBy off entirely removes the clock they want',
     plan=['swipe up', 'tap StandBy', 'tap Motion to Wake', 'tap Show Notifications', 'press back', 'tap Display & Brightness', 'swipe up', 'tap Raise to Wake'],
     goal=g_paths({'settings.raise_wake': False, 'settings.spec.standby.motion': False, 'settings.spec.standby.show_notif': False}))

task(id='notification_triage', family='notifications', band=3,
     instruction="Only actual people should be allowed to interrupt me. Mail can keep arriving but silently and without a count on the icon, and the news and the podcasts shouldn't notify me at all any more.",
     start_app='settings', seed=0,
     mirrors='Apple Support: Notification settings per app',
     trap='the three apps have separate notification pages; silencing mail entirely loses what is wanted',
     plan=['swipe up', 'tap Notifications', 'swipe up', 'tap Mail', 'tap Sounds', 'tap Badges', 'press back', 'press back', 'swipe up', 'swipe up', 'tap Apps', 'swipe up', 'tap News', 'tap Notifications', 'tap Allow Notifications', 'press back', 'press back', 'tap Podcasts', 'tap Notifications', 'tap Allow Notifications'],
     goal=g_paths({'settings.apps.Mail.notif.sounds': False, 'settings.apps.Mail.notif.badges': False, 'settings.apps.Podcasts.notif.allow': False, 'settings.apps.News.notif.allow': False}))

task(id='pay_friend_no_fee', family='payments', band=3,
     instruction="I owe Alex Kim a hundred and twenty for the concert tickets. Send it from my QuickPay balance, and I'm not paying a fee for it — if it takes a few days to land, fine. Put a note on it so he knows what it's for.",
     start_app='bank', seed=0,
     mirrors='QuickPay: send money with a note',
     trap='the instant option is preselected and charges a fee; the card below the balance charges another one',
     plan=['tap Send money', 'tap Alex Kim', 'tap the amount box', 'type "120"', 'press back', 'tap the Note field', 'type "concert tickets"', 'press back', 'tap 1-3 business days', 'swipe up', 'tap Review', 'tap Confirm & send', 'tap the code field', 'type OTP', 'tap Verify & send'],
     goal=g_paths({'bank.balance': 5164.55, 'bank.otp.code': '624684', 'bank.otp.verified': True, 'bank.history': [{'id': 1, 'amount': 120.0, 'payee': 'Alex Kim', 'handle': '@akim', 'source': 'balance', 'speed': 'standard', 'note': 'concert tickets'}], 'messages.threads': [{'id': 'qp', 'name': 'QuickPay', 'handle': '22395', 'msgs': [{'from': 'them', 'text': 'Your QuickPay code is 624684. Expires in 30s.'}], 'unread': True}, {'id': 'mom', 'name': 'Mom', 'handle': '+1 555 0100', 'time': '9:02', 'msgs': [{'from': 'them', 'text': 'Are you coming Sunday?'}], 'unread': False}, {'id': 'alex', 'name': 'Alex Chen', 'handle': '@alexc', 'time': '9:31', 'msgs': [{'from': 'them', 'text': 'did you send the $40 yet?'}], 'unread': True}]}))

task(id='photos_private', family='privacy', band=3,
     instruction='My photo library should stop sharing albums with anyone, stop using mobile data, and stop notifying me about memories and the rest of it.',
     start_app='settings', seed=0,
     mirrors='Apple Support: Shared Albums; app notifications',
     trap="'Shared Library' is a row that looks like the sharing switch and is not it",
     plan=['swipe up', 'swipe up', 'swipe up', 'tap Apps', 'swipe up', 'tap Photos', 'tap Shared Albums', 'tap Background App Refresh', 'tap Notifications', 'tap Allow Notifications'],
     goal=g_paths({'settings.apps.Photos.bg': False, 'settings.apps.Photos.notif.allow': False, 'settings.spec.appx.photos_shared_albums': False}))

task(id='quickpay_location_and_notifications', family='privacy', band=3,
     instruction="The payments app is too nosy. It shouldn't know where I am at all, and it shouldn't be allowed to put anything on my screen or make any noise. Leave every other app's notifications exactly as they are.",
     start_app='settings', seed=0,
     mirrors='Apple Support: App permissions; notification settings',
     trap='its location page and its notification page are in different parts of Settings',
     plan=['swipe up', 'swipe up', 'tap Privacy & Security', 'tap Location Services', 'tap QuickPay', 'tap Never', 'press back', 'press back', 'press back', 'tap Notifications', 'tap QuickPay', 'tap Allow Notifications'],
     goal=g_paths({'settings.apps.QuickPay.loc_mode': 'Never', 'settings.apps.QuickPay.notif.allow': False}))

task(id='quickpay_permission_audit', family='privacy', band=3,
     instruction="QuickPay is for paying people. It shouldn't be able to listen through my microphone, look through my camera, or read my address book. Take all three away from it and leave every other app exactly as it is.",
     start_app='settings', seed=0,
     mirrors='Apple Support: Control app access to microphone, camera and contacts',
     trap="the three permissions live on three different pages; the app's own page only shows two of them",
     plan=['swipe up', 'swipe up', 'tap Privacy & Security', 'tap Microphone', 'tap QuickPay', 'press back', 'tap Camera', 'tap QuickPay', 'press back', 'press back', 'swipe up', 'tap Apps', 'swipe up', 'tap QuickPay', 'tap Contacts'],
     goal=g_paths({'settings.apps.QuickPay.perms.Camera': False, 'settings.apps.QuickPay.perms.Microphone': False, 'settings.apps.QuickPay.perms.Contacts': False}))

task(id='standby_bedside_dark', family='display', band=3,
     instruction="When the phone is docked on my nightstand overnight, the clock should go dark on its own instead of glowing all night, and it shouldn't light up because I rolled over. I do still want the clock there when I look at it.",
     start_app='settings', seed=0,
     mirrors='Apple Support: StandBy display options',
     trap="'Never' for the display and turning StandBy off both remove the clock they want kept",
     plan=['swipe up', 'tap StandBy', 'tap Motion to Wake', 'tap Display', 'tap Always On', 'tap Turn Display Off', 'tap After 20 Seconds'],
     goal=g_paths({'settings.spec.standby.display_off': 'After 20 Seconds', 'settings.spec.standby.always_on': False, 'settings.spec.standby.motion': False}))

task(id='stolen_device_guard', family='security', band=3,
     instruction="If a pickpocket watches me type my passcode and then takes the phone, they shouldn't be able to change my account or wipe it. Switch on the protection that puts a delay on those when I'm somewhere unfamiliar.",
     start_app='settings', seed=0,
     mirrors='Apple Support: Stolen Device Protection',
     trap='it lives at the bottom of the passcode page, under the lock-screen switches people stop at',
     plan=['swipe up', 'swipe up', 'tap Face ID & Passcode', 'swipe up', 'tap Stolen Device Protection', 'tap Stolen Device Protection'],
     goal=g_paths({'settings.spec.faceid.sdp': True}))

task(id='stranger_shield', family='privacy', band=3,
     instruction="I'm drowning in spam. Calls from numbers that aren't in my contacts shouldn't ring at all, the same for video calls, and texts from people I don't know should be sorted out of my main list instead of landing in it.",
     start_app='settings', seed=0,
     mirrors='Apple Support: Silence unknown callers; Filter unknown senders',
     trap='the three switches live in three different app pages; blocking a contact does nothing here',
     plan=['swipe up', 'swipe up', 'swipe up', 'tap Apps', 'swipe up', 'tap Phone', 'tap Silence Unknown Callers', 'press back', 'swipe up', 'swipe up', 'swipe up', 'swipe up', 'swipe up', 'swipe up', 'swipe up', 'swipe up', 'swipe down', 'swipe down', 'tap FaceTime', 'tap Silence Unknown Callers', 'press back', 'tap Messages', 'swipe up', 'tap Filter Unknown Senders'],
     goal=g_paths({'settings.spec.appx.msg_filter': True, 'settings.spec.appx.phone_silence_unknown': True, 'settings.spec.appx.ft_silence_unknown': True}))

task(id='travel_data_diet', family='cellular', band=3,
     instruction='I land in Lisbon tomorrow and data there costs a fortune. Only the browser and the maps app may use mobile data — everything else waits for wi-fi. I still need mobile data itself switched on for those two.',
     start_app='settings', seed=0,
     mirrors='Apple Support: Set cellular data options per app',
     trap='switching off Cellular Data at the top kills the two apps that must keep working',
     plan=['tap Cellular', 'tap TripSnap', 'tap QuickPay', 'tap Weatherly', 'swipe up', 'tap Music'],
     goal=g_paths({'settings.apps.TripSnap.cellular': False, 'settings.apps.QuickPay.cellular': False, 'settings.apps.Weatherly.cellular': False, 'settings.apps.Music.cellular': False}))

task(id='voice_control_numbers_only', family='accessibility', band=3,
     instruction='I drive my phone by voice. Turn that on, and when it labels things on screen I want the numbers only — not the names, not the grid over everything.',
     start_app='settings', seed=0,
     mirrors='Apple Support: Voice Control overlays',
     trap='the overlay choices are on a page two levels below the switch that turns it on',
     plan=['tap Accessibility', 'swipe up', 'tap Voice Control', 'tap Voice Control', 'tap Commands', 'tap Overlays', 'tap Show Names', 'tap Show Grid'],
     goal=g_paths({'settings.spec.accessibility.voice_control': True, 'settings.spec.gen.overlays.show_grid': False, 'settings.spec.gen.overlays.show_names': False}))

task(id='grandparent_setup_full', family='accessibility', band=4,
     instruction="I'm setting this phone up for my grandmother before she flies out. She can't read thin grey text, she can't tell which switches are on, she can't see where a button ends and the background starts, she gets lost when things slide around, and the screen keeps locking while she's still reading it — give her five minutes before it does. Don't invert the colours, and don't stop it locking altogether.",
     start_app='settings', seed=0,
     mirrors='Apple Support: Display & Text Size; Reduce Motion; Auto-Lock',
     trap="Smart Invert answers 'can't read it'; 'Never' answers the locking complaint and leaves the phone open forever",
     plan=['tap Accessibility', 'tap Display & Text Size', 'tap Bold Text', 'tap Increase Contrast', 'tap On/Off Labels', 'tap Button Shapes', 'press back', 'tap Motion', 'tap Reduce Motion', 'press back', 'press back', 'swipe up', 'tap Display & Brightness', 'tap Auto-Lock', 'tap 5 Minutes'],
     goal=g_paths({'settings.auto_lock': '5 Minutes', 'settings.spec.accessibility.bold': True, 'settings.spec.accessibility.reduce_motion': True, 'settings.spec.accessibility.increase_contrast': True, 'settings.spec.accessibility.button_shapes': True, 'settings.spec.accessibility.on_off_labels': True}))

task(id='handing_phone_over', family='privacy', band=4,
     instruction="I'm lending this phone to a colleague for the afternoon. Nothing of mine should be reachable from the locked screen — no notifications, no replying to my messages, no paying with my cards, no calling people back, and no widgets. It should also demand the passcode the moment the screen goes dark. Face ID must still unlock it for me afterwards.",
     start_app='settings', seed=0,
     mirrors='Apple Support: Allow access when locked; Require passcode',
     trap='turning iPhone Unlock off locks the owner out; Today View is the widget row people miss',
     plan=['swipe up', 'swipe up', 'tap Face ID & Passcode', 'tap Allow Access When Locked', 'tap Notification Center', 'tap Reply with Message', 'tap Wallet', 'tap Return Missed Calls', 'tap Lock Screen Widgets', 'tap Today View and Search', 'press back', 'tap Require Passcode', 'tap Immediately'],
     goal=g_paths({'settings.spec.lockaccess.lock_screen_widgets': False, 'settings.spec.lockaccess.today': False, 'settings.spec.lockaccess.notif_center': False, 'settings.spec.lockaccess.lock_reply': False, 'settings.spec.lockaccess.wallet': False, 'settings.spec.lockaccess.return_missed': False}))

task(id='loaner_phone_setup', family='privacy', band=4,
     instruction="I'm lending this to a contractor for a week. He shouldn't be able to reach the web or the camera, the assistant shouldn't search the web for him or read out anything with swearing in it, nothing should arrive from strangers standing nearby, and the news app shouldn't bother him at all.",
     start_app='settings', seed=0,
     mirrors='Apple Support: Content & Privacy Restrictions; AirDrop; notifications',
     trap="'Contacts Only' still accepts from people in the address book",
     plan=['swipe up', 'swipe up', 'tap Screen Time', 'tap Content & Privacy Restrictions', 'tap Allowed Apps & Features', 'tap Safari', 'tap Camera', 'press back', 'tap Intelligence & Siri', 'tap Web Search Content', 'tap Explicit Language', 'press back', 'press back', 'press back', 'swipe up', 'swipe up', 'swipe up', 'swipe up', 'swipe up', 'swipe up', 'swipe up', 'swipe up', 'swipe down', 'tap General', 'tap AirDrop', 'tap Receiving Off', 'press back', 'press back', 'swipe up', 'tap Apps', 'swipe up', 'tap News', 'tap Notifications', 'tap Allow Notifications'],
     goal=g_paths({'settings.apps.News.notif.allow': False, 'settings.spec.airdrop': 'Receiving Off', 'settings.spec.gen.allowed_apps_features.camera': False, 'settings.spec.gen.allowed_apps_features.safari': False, 'settings.spec.gen.intelligence_siri.explicit_language': False, 'settings.spec.gen.intelligence_siri.web_search_content': False}))

task(id='lost_phone_drill', family='security', band=4,
     instruction="I'm going somewhere with a lot of pickpockets. If someone lifts this phone, they shouldn't be able to read my notifications, answer my messages, pay with my cards or call anyone back — none of that from the locked screen. It should want the passcode the instant the screen goes dark, and it should wipe itself if someone sits there guessing the passcode. Face ID still has to unlock it for me.",
     start_app='settings', seed=0,
     mirrors='Apple Support: Allow access when locked; Require passcode; Erase data',
     trap="turning iPhone Unlock off locks the owner out and is the tempting 'make it secure' move",
     plan=['swipe up', 'swipe up', 'tap Face ID & Passcode', 'tap Allow Access When Locked', 'tap Notification Center', 'tap Reply with Message', 'tap Wallet', 'tap Return Missed Calls', 'press back', 'tap Require Passcode', 'tap Immediately', 'press back', 'swipe up', 'tap Erase Data'],
     goal=g_paths({'settings.spec.faceid.erase_data': True, 'settings.spec.lockaccess.notif_center': False, 'settings.spec.lockaccess.lock_reply': False, 'settings.spec.lockaccess.wallet': False, 'settings.spec.lockaccess.return_missed': False}))

task(id='low_vision_full_setup', family='accessibility', band=4,
     instruction="My father is losing his sight. Make the text heavy and high-contrast, mark the switches and the buttons so he can see what is tappable, cut the see-through layers that wash everything out, give him the screen magnifier that follows what he's typing, and stop the screen locking on him after half a minute — five minutes is about right. Don't invert the colours.",
     start_app='settings', seed=0,
     mirrors='Apple Support: Display & Text Size; Zoom; Auto-Lock',
     trap="Smart Invert and Classic Invert are the tempting answers and are explicitly refused; 'Never' for locking leaves the phone permanently open",
     plan=['tap Accessibility', 'tap Display & Text Size', 'tap Bold Text', 'tap Increase Contrast', 'tap On/Off Labels', 'tap Button Shapes', 'tap Reduce Transparency', 'press back', 'tap Zoom', 'tap Zoom', 'tap Follow Focus', 'tap Smart Typing', 'press back', 'press back', 'swipe up', 'tap Display & Brightness', 'tap Auto-Lock', 'tap 5 Minutes'],
     goal=g_paths({'settings.auto_lock': '5 Minutes', 'settings.spec.accessibility.bold': True, 'settings.spec.accessibility.reduce_transparency': True, 'settings.spec.accessibility.increase_contrast': True, 'settings.spec.accessibility.button_shapes': True, 'settings.spec.accessibility.on_off_labels': True, 'settings.spec.accessibility.follow_focus': True, 'settings.spec.accessibility.smart_typing': True, 'settings.spec.accessibility.zoom': True}))

task(id='motor_impairment_setup', family='accessibility', band=4,
     instruction='My hands barely work any more. Put the on-screen button on the display that can press the hardware buttons for me and make it confirm before acting, turn on the scanning control so I can drive the phone with a switch, and stop the trackpad gestures that I keep triggering by accident — the pinch and the rotate ones.',
     start_app='settings', seed=0,
     mirrors='Apple Support: AssistiveTouch; Switch Control; Full Keyboard Access',
     trap="Switch Control's own sub-switches look like the ones to change and are not",
     plan=['tap Accessibility', 'tap Touch', 'tap AssistiveTouch', 'tap AssistiveTouch', 'tap Confirm with AssistiveTouch', 'press back', 'press back', 'tap Switch Control', 'tap Switch Control', 'press back', 'swipe up', 'tap Keyboards & Typing', 'tap Full Keyboard Access', 'tap Commands', 'tap Hand Gestures', 'tap Pinch', 'tap Rotate'],
     goal=g_paths({'settings.spec.acc_sw.on': True, 'settings.spec.acc_at.on': True, 'settings.spec.acc_at.confirm': True, 'settings.spec.gen.hand_gestures.pinch': False, 'settings.spec.gen.hand_gestures.rotate': False}))

task(id='mum_new_phone', family='accessibility', band=4,
     instruction="I'm setting my mother's new phone up. She can't read thin text and can't tell which switches are on, the screen locks before she's finished reading it — five minutes suits her — strangers keep sending her things she doesn't understand so nothing should arrive from anyone nearby, and the news app frightens her with alerts so it shouldn't notify her at all.",
     start_app='settings', seed=0,
     mirrors='Apple Support: Display & Text Size; Auto-Lock; AirDrop; notifications',
     trap="'Contacts Only' still lets things through; 'Never' for the screen lock leaves the phone open on a bus",
     plan=['tap Accessibility', 'tap Display & Text Size', 'tap Bold Text', 'tap Increase Contrast', 'tap On/Off Labels', 'press back', 'press back', 'swipe up', 'tap Display & Brightness', 'tap Auto-Lock', 'tap 5 Minutes', 'press back', 'press back', 'tap General', 'tap AirDrop', 'tap Receiving Off', 'press back', 'press back', 'swipe up', 'swipe up', 'tap Apps', 'swipe up', 'tap News', 'tap Notifications', 'tap Allow Notifications'],
     goal=g_paths({'settings.auto_lock': '5 Minutes', 'settings.apps.News.notif.allow': False, 'settings.spec.airdrop': 'Receiving Off', 'settings.spec.accessibility.bold': True, 'settings.spec.accessibility.increase_contrast': True, 'settings.spec.accessibility.on_off_labels': True}))

task(id='pay_landlord_today', family='payments', band=4,
     instruction="Rent is late. Send the landlord nine hundred and fifty, label it October rent, and it has to land today — I'll wear the fee. Take it off my Visa, not my QuickPay balance; I need that balance for the deposit.",
     start_app='bank', seed=0,
     mirrors='QuickPay: choose a funding source and speed',
     trap='the balance is preselected and is exactly what must not be used; the no-fee speed would miss the deadline',
     plan=['tap Send money', 'tap Landlord — Sunset Apts', 'tap the amount box', 'type "950"', 'press back', 'tap the Note field', 'type "October rent"', 'press back', 'tap Visa ···· 4821 · 3% fee', 'swipe up', 'tap Review', 'tap Confirm & send', 'tap the code field', 'type OTP', 'tap Verify & send'],
     goal=g_paths({'bank.balance': 4334.55, 'bank.otp.code': '819684', 'bank.otp.verified': True, 'bank.history': [{'id': 1, 'amount': 950.0, 'payee': 'Landlord — Sunset Apts', 'handle': '@sunsetapts', 'source': 'card', 'speed': 'instant', 'note': 'October rent'}], 'messages.threads': [{'id': 'qp', 'name': 'QuickPay', 'handle': '22395', 'msgs': [{'from': 'them', 'text': 'Your QuickPay code is 819684. Expires in 30s.'}], 'unread': True}, {'id': 'mom', 'name': 'Mom', 'handle': '+1 555 0100', 'time': '9:02', 'msgs': [{'from': 'them', 'text': 'Are you coming Sunday?'}], 'unread': False}, {'id': 'alex', 'name': 'Alex Chen', 'handle': '@alexc', 'time': '9:31', 'msgs': [{'from': 'them', 'text': 'did you send the $40 yet?'}], 'unread': True}]}))

task(id='payments_need_unlock', family='payments', band=4,
     instruction="Nobody should be able to spend money with this phone without unlocking it first. No paying on a turnstile, no shortcut on the side button, no cards reachable from the locked screen, and face recognition shouldn't authorise payments either. Face recognition can still unlock the phone itself.",
     start_app='settings', seed=0,
     mirrors='Apple Support: Wallet & Apple Pay; Allow access when locked; Face ID',
     trap='turning iPhone Unlock off is the heavy-handed version and locks the owner out',
     plan=['swipe up', 'swipe up', 'swipe up', 'tap Wallet & Apple Pay', 'tap Express Transit', 'tap Double-Click Side Button', 'press back', 'tap Face ID & Passcode', 'tap Allow Access When Locked', 'tap Wallet', 'press back', 'tap Wallet & Apple Pay'],
     goal=g_paths({'settings.spec.faceid.wallet': False, 'settings.spec.lockaccess.wallet': False, 'settings.spec.wallet.double_click': False, 'settings.spec.wallet.express': False}))

task(id='privacy_deep_clean', family='privacy', band=4,
     instruction="I want to stop being profiled. No app should even be able to ask to follow me across other companies' apps, Apple should stop collecting how I use this phone and stop keeping recordings of what I say to the assistant — but keep sending the iCloud usage data, my work needs that. On top of that, the travel diary should never get my location again and the weather app should have to ask each time, with only the rough area.",
     start_app='settings', seed=0,
     mirrors='Apple Support: Tracking; Analytics & Improvements; Location Services',
     trap='the iCloud analytics switch sits between the two that must go off and must stay on',
     plan=['swipe up', 'swipe up', 'tap Privacy & Security', 'tap Tracking', 'tap Allow Apps to Request to Track', 'press back', 'swipe up', 'tap Analytics & Improvements', 'tap Share iPhone Analytics', 'tap Improve Siri & Dictation', 'press back', 'swipe up', 'swipe up', 'swipe up', 'swipe up', 'swipe up', 'swipe up', 'swipe up', 'swipe up', 'swipe down', 'tap Location Services', 'tap TripSnap', 'tap Never', 'press back', 'tap Weatherly', 'tap Ask Next Time Or When I Share', 'tap Precise Location'],
     goal=g_paths({'settings.apps.TripSnap.loc_mode': 'Never', 'settings.apps.Weatherly.loc_mode': 'Ask Next Time Or When I Share', 'settings.apps.Weatherly.precise': False, 'settings.spec.priv.tracking': False, 'settings.spec.priv.improve_siri_dictation': False, 'settings.spec.priv.analytics': False}))

task(id='roaming_full_lockdown', family='cellular', band=4,
     instruction="I'm abroad for a fortnight and roaming is extortionate. Only the browser and the maps may touch mobile data, nothing may quietly refresh itself in the background except my calendar. Mobile data itself stays on — I need those two apps working.",
     start_app='settings', seed=0,
     mirrors='Apple Support: Cellular data per app; Background App Refresh',
     trap='switching Cellular Data off at the top is the one move that breaks what must keep working',
     plan=['tap Cellular', 'tap TripSnap', 'tap QuickPay', 'tap Weatherly', 'swipe up', 'tap Music', 'press back', 'tap General', 'tap Background App Refresh', 'tap TripSnap', 'tap QuickPay', 'tap Weatherly', 'tap Safari', 'tap Maps', 'tap Music'],
     goal=g_paths({'settings.apps.TripSnap.cellular': False, 'settings.apps.TripSnap.bg': False, 'settings.apps.QuickPay.cellular': False, 'settings.apps.QuickPay.bg': False, 'settings.apps.Weatherly.cellular': False, 'settings.apps.Weatherly.bg': False, 'settings.apps.Safari.bg': False, 'settings.apps.Maps.bg': False, 'settings.apps.Music.cellular': False, 'settings.apps.Music.bg': False}))
