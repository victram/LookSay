# Manual usability tests

Run these with the real hardware and, before claiming the app is suitable
for daily use, with people who have motor disabilities and their caregivers.
For each test, note the webcam model, lighting, glasses, distance to the
screen, and the results (time, errors, comments).

| # | Scenario | Steps | Pass criteria |
|---|---|---|---|
| 1 | First calibration | Fresh start → Settings ▸ Calibrate (9 points) → follow the dots → Accept by gaze | Finishes in under 30 s; the reported error is shown; the user can select "Accept" by gaze |
| 2 | 5-point calibration | Settings ▸ Eye tracking ▸ Calibration points = 5 → Calibrate | Works; the gaze cursor follows the gaze over all 4 quadrants |
| 3 | Typing | Type "I need water" with letters and suggestions only | No physical input; ≤ 2 wrong selections; record the time |
| 4 | Corrections | Make a typo → ⌫, Del word, Undo | Each control does exactly one step |
| 5 | Word mode | Switch to Words, compose "I want to sleep" | Words insert with correct spacing and capitals |
| 6 | Speech | Speak, Stop mid-sentence; Speak mode = Last sentence | Audio starts and stops; only the last sentence is read |
| 7 | Emergency | Select HELP from the keyboard and the first phrase on the phrases page | The emergency phrase is spoken at full volume within the dwell time; nothing is contacted automatically |
| 8 | Face loss | Turn away or cover the camera while dwelling on a key | The status shows "Face not found"; no selection happens; tracking resumes when the face returns |
| 9 | Poor lighting | Dim light / strong backlight | Confidence drops and selection pauses rather than selecting wrong keys |
| 10 | Glasses | Repeat 3 with glasses | Note the accuracy difference |
| 11 | Drift | Use for 10 min, shift posture → Recenter | Accuracy recovers after Recenter |
| 12 | Blink to select | Enable it → deliberate blink (~0.5–1.5 s) on a key | Selects; natural blinks and resting the eyes do not |
| 13 | Pause | Pause → look around for 1 min → Resume | Nothing is selected while paused |
| 14 | Settings by gaze | Change dwell time, theme, language and text size by gaze only | All work; no mouse needed |
| 15 | Languages | EN → ES → CA with text in the canvas | The text is kept; layouts (ñ, ç, ·), labels and voices change |
| 16 | Restart persistence | Change settings, save a phrase, type learned words → quit → restart | Everything is kept; calibration is still active |
| 17 | Camera unplugged | Unplug the webcam during use, then plug it back in | The status says the camera is unavailable; the mouse still works; tracking resumes |
| 18 | Camera denied | Deny camera permission (Windows privacy settings) | The app still starts; the mouse works; the status explains the problem |
| 19 | Export / import / reset | Export → Delete all data (confirm) → Import | Settings and phrases come back; learned words and calibration are gone, as expected |
| 20 | Fatigue | 20 minutes of continuous use | The user's comfort rating; any accidental selections |
