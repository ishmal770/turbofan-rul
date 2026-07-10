"""
Choreo Builder - Pygame UI demo (with home screen + menu)
----------------------------------------------------------
Screens:
  HOME   - logo + "Enter" button
  MENU   - 3 options: Pose Tracker / Choreography Tracker / Editing Position Tracker
  EDIT   - drag circles into formation, save keyframes (the "Editing Position Tracker")
  REPLAY - watch saved keyframes animate in sync with a fake clock (the "Choreography Tracker")
  POSE   - placeholder screen for the webcam + MediaPipe pose comparison (not built yet)

Run with:  python choreo_demo.py
Requires:  pip install pygame
"""

import pygame
import json
import sys

pygame.init()

WIDTH, HEIGHT = 900, 650
CANVAS_HEIGHT = 480
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Choreo Builder - Demo")

font = pygame.font.SysFont("arial", 16)
title_font = pygame.font.SysFont("arial", 20, bold=True)
logo_font = pygame.font.SysFont("arial", 44, bold=True)
subtitle_font = pygame.font.SysFont("arial", 18)

BG = (12, 12, 12)
CANVAS_BG = (22, 22, 22)
WHITE = (235, 235, 235)
GRAY = (70, 70, 70)
LIGHT_GRAY = (150, 150, 150)
GREEN = (90, 200, 130)
RED = (220, 90, 90)
BLUE = (90, 140, 220)
PURPLE = (150, 100, 210)
AMBER = (220, 165, 70)

# ---- App state ------------------------------------------------------

SCREEN = "home"   # "home" | "menu" | "edit" | "replay" | "pose"

dancers = [
    {"id": "d1", "name": "Aarush", "x": 300, "y": 250, "color": (226, 75, 74)},
    {"id": "d2", "name": "Priya",  "x": 600, "y": 250, "color": (55, 138, 222)},
    {"id": "d3", "name": "Zara",   "x": 450, "y": 380, "color": (99, 153, 34)},
]

keyframes = []  # each: {"time": float, "positions": {dancer_id: {"x":0-1, "y":0-1}}}

dragging = None
drag_offset = (0, 0)

song_length = 20.0
current_time = 0.0
playing = False

clock = pygame.time.Clock()

# ---- Shared helpers ---------------------------------------------------


def draw_button(rect, text, color, text_color=WHITE):
    pygame.draw.rect(screen, color, rect, border_radius=10)
    label = font.render(text, True, text_color)
    screen.blit(label, (rect.x + (rect.width - label.get_width()) // 2,
                        rect.y + (rect.height - label.get_height()) // 2))


def draw_back_button():
    rect = pygame.Rect(20, 20, 100, 36)
    draw_button(rect, "< Back", GRAY)
    return rect


def save_keyframe():
    positions = {d["id"]: {"x": d["x"] / WIDTH, "y": d["y"] / CANVAS_HEIGHT} for d in dancers}
    kf = {"time": round(current_time, 2), "positions": positions}
    keyframes[:] = [k for k in keyframes if abs(k["time"] - kf["time"]) > 0.15]
    keyframes.append(kf)
    keyframes.sort(key=lambda k: k["time"])
    with open("choreo.json", "w") as f:
        json.dump({"dancers": dancers, "keyframes": keyframes}, f, indent=2)
    print(f"Saved keyframe at t={kf['time']}s  ({len(keyframes)} total)")


def interpolate_positions(t):
    if not keyframes:
        return None
    if t <= keyframes[0]["time"]:
        return keyframes[0]["positions"]
    if t >= keyframes[-1]["time"]:
        return keyframes[-1]["positions"]
    for i in range(len(keyframes) - 1):
        k1, k2 = keyframes[i], keyframes[i + 1]
        if k1["time"] <= t <= k2["time"]:
            span = k2["time"] - k1["time"]
            frac = 0 if span == 0 else (t - k1["time"]) / span
            result = {}
            for d in dancers:
                did = d["id"]
                p1 = k1["positions"].get(did, {"x": 0.5, "y": 0.5})
                p2 = k2["positions"].get(did, {"x": 0.5, "y": 0.5})
                result[did] = {
                    "x": p1["x"] + (p2["x"] - p1["x"]) * frac,
                    "y": p1["y"] + (p2["y"] - p1["y"]) * frac,
                }
            return result
    return keyframes[-1]["positions"]


# ---- Screen: HOME ------------------------------------------------------

enter_btn = pygame.Rect(WIDTH // 2 - 90, 400, 180, 50)


def draw_home():
    screen.fill(BG)
    # simple drawn logo: a ring of dots around a circle, stands in for a real logo image
    cx, cy = WIDTH // 2, 220
    pygame.draw.circle(screen, PURPLE, (cx, cy), 70, width=4)
    for i in range(8):
        import math
        ang = i * (360 / 8)
        px = cx + 90 * math.cos(math.radians(ang))
        py = cy + 90 * math.sin(math.radians(ang))
        pygame.draw.circle(screen, (BLUE if i % 2 == 0 else GREEN), (int(px), int(py)), 8)

    logo_text = logo_font.render("ChoreoAI", True, WHITE)
    screen.blit(logo_text, (WIDTH // 2 - logo_text.get_width() // 2, cy + 110))

    sub = subtitle_font.render("Track. Compare. Correct. Replay.", True, LIGHT_GRAY)
    screen.blit(sub, (WIDTH // 2 - sub.get_width() // 2, cy + 165))

    draw_button(enter_btn, "Enter", GREEN)


def handle_home_click(pos):
    global SCREEN
    if enter_btn.collidepoint(pos):
        SCREEN = "menu"


# ---- Screen: MENU ------------------------------------------------------

menu_options = [
    {"key": "pose",   "label": "Pose Tracker",              "color": AMBER,
     "desc": "Live webcam pose comparison"},
    {"key": "replay", "label": "Choreography Tracker",      "color": BLUE,
     "desc": "Watch the full routine play back"},
    {"key": "edit",   "label": "Editing Position Tracker",  "color": PURPLE,
     "desc": "Place dancers and save formations"},
]

menu_buttons = []
btn_w, btn_h, gap = 220, 160, 30
total_w = len(menu_options) * btn_w + (len(menu_options) - 1) * gap
start_x = (WIDTH - total_w) // 2
for i, opt in enumerate(menu_options):
    rect = pygame.Rect(start_x + i * (btn_w + gap), 230, btn_w, btn_h)
    menu_buttons.append(rect)


def draw_menu():
    screen.fill(BG)
    header = title_font.render("Choose a mode", True, WHITE)
    screen.blit(header, (WIDTH // 2 - header.get_width() // 2, 140))

    for rect, opt in zip(menu_buttons, menu_options):
        pygame.draw.rect(screen, opt["color"], rect, border_radius=14)
        label = title_font.render(opt["label"], True, WHITE)
        # wrap label onto two lines if needed by splitting on spaces roughly
        screen.blit(label, (rect.x + rect.width // 2 - label.get_width() // 2, rect.y + 45))
        desc = font.render(opt["desc"], True, (30, 30, 30))
        screen.blit(desc, (rect.x + rect.width // 2 - desc.get_width() // 2, rect.y + 95))


def handle_menu_click(pos):
    global SCREEN
    for rect, opt in zip(menu_buttons, menu_options):
        if rect.collidepoint(pos):
            SCREEN = opt["key"]


# ---- Screen: EDIT (Editing Position Tracker) ---------------------------

save_btn = pygame.Rect(140, 560, 170, 40)
scrub_hint_y = CANVAS_HEIGHT + 20


def draw_edit():
    screen.fill(BG)
    pygame.draw.rect(screen, CANVAS_BG, (0, 60, WIDTH, CANVAS_HEIGHT - 60))

    header = title_font.render("Editing Position Tracker", True, WHITE)
    screen.blit(header, (140, 20))

    for d in dancers:
        pygame.draw.circle(screen, d["color"], (int(d["x"]), int(d["y"])), 30)
        label = font.render(d["name"], True, WHITE)
        screen.blit(label, (d["x"] - label.get_width() // 2, d["y"] - 8))

    timeline_y = scrub_hint_y
    pygame.draw.rect(screen, GRAY, (20, timeline_y, WIDTH - 40, 10), border_radius=5)
    for kf in keyframes:
        kx = 20 + (kf["time"] / song_length) * (WIDTH - 40)
        pygame.draw.circle(screen, GREEN, (int(kx), timeline_y + 5), 6)
    playhead_x = 20 + (current_time / song_length) * (WIDTH - 40)
    pygame.draw.line(screen, RED, (playhead_x, timeline_y - 10), (playhead_x, timeline_y + 20), 2)

    time_label = font.render(
        f"t = {current_time:.1f}s / {song_length:.0f}s    keyframes saved: {len(keyframes)}",
        True, WHITE)
    screen.blit(time_label, (20, timeline_y + 25))

    hint = font.render("Drag circles, then Save Keyframe. Click the gray bar to scrub time.",
                        True, LIGHT_GRAY)
    screen.blit(hint, (20, timeline_y + 45))

    draw_button(save_btn, "Save Keyframe", GREEN)
    draw_back_button()


def handle_edit_mousedown(pos):
    global dragging, drag_offset, current_time
    mx, my = pos
    if pygame.Rect(20, 20, 100, 36).collidepoint(pos):
        return "back"
    if save_btn.collidepoint(pos):
        save_keyframe()
        return None
    if scrub_hint_y - 5 < my < scrub_hint_y + 15:
        frac = max(0, min(1, (mx - 20) / (WIDTH - 40)))
        current_time = frac * song_length
        return None
    if 60 < my < CANVAS_HEIGHT:
        for d in dancers:
            if (mx - d["x"]) ** 2 + (my - d["y"]) ** 2 < 32 ** 2:
                dragging = d
                drag_offset = (mx - d["x"], my - d["y"])
    return None


# ---- Screen: REPLAY (Choreography Tracker) -----------------------------

play_btn = pygame.Rect(WIDTH // 2 - 85, 560, 170, 40)


def draw_replay():
    global current_time, playing
    screen.fill(BG)
    pygame.draw.rect(screen, CANVAS_BG, (0, 60, WIDTH, CANVAS_HEIGHT - 60))

    header = title_font.render("Choreography Tracker (Replay)", True, WHITE)
    screen.blit(header, (140, 20))

    if playing:
        current_time += clock.get_time() / 1000.0
        if current_time >= song_length:
            current_time = song_length
            playing = False
        positions = interpolate_positions(current_time)
        if positions:
            for d in dancers:
                p = positions.get(d["id"])
                if p:
                    d["x"] = p["x"] * WIDTH
                    d["y"] = p["y"] * CANVAS_HEIGHT

    for d in dancers:
        pygame.draw.circle(screen, d["color"], (int(d["x"]), int(d["y"])), 30)
        label = font.render(d["name"], True, WHITE)
        screen.blit(label, (d["x"] - label.get_width() // 2, d["y"] - 8))

    timeline_y = scrub_hint_y
    pygame.draw.rect(screen, GRAY, (20, timeline_y, WIDTH - 40, 10), border_radius=5)
    for kf in keyframes:
        kx = 20 + (kf["time"] / song_length) * (WIDTH - 40)
        pygame.draw.circle(screen, GREEN, (int(kx), timeline_y + 5), 6)
    playhead_x = 20 + (current_time / song_length) * (WIDTH - 40)
    pygame.draw.line(screen, RED, (playhead_x, timeline_y - 10), (playhead_x, timeline_y + 20), 2)

    time_label = font.render(f"t = {current_time:.1f}s / {song_length:.0f}s", True, WHITE)
    screen.blit(time_label, (20, timeline_y + 25))

    if not keyframes:
        warn = font.render("No keyframes saved yet - go save some in Editing Position Tracker first.",
                            True, AMBER)
        screen.blit(warn, (20, timeline_y + 45))

    draw_button(play_btn, "Pause" if playing else "Play", BLUE)
    draw_back_button()


def handle_replay_mousedown(pos):
    global playing, current_time
    if pygame.Rect(20, 20, 100, 36).collidepoint(pos):
        return "back"
    if play_btn.collidepoint(pos):
        playing = not playing
    return None


# ---- Screen: POSE (placeholder) ----------------------------------------

def draw_pose_stub():
    screen.fill(BG)
    header = title_font.render("Pose Tracker", True, WHITE)
    screen.blit(header, (140, 20))

    msg1 = subtitle_font.render("Webcam + live pose comparison goes here.", True, WHITE)
    msg2 = font.render("(Not wired up yet - this is where MediaPipe pose detection",
                        True, LIGHT_GRAY)
    msg3 = font.render("will compare the dancer's live position/pose to the nearest keyframe.)",
                        True, LIGHT_GRAY)
    screen.blit(msg1, (WIDTH // 2 - msg1.get_width() // 2, 260))
    screen.blit(msg2, (WIDTH // 2 - msg2.get_width() // 2, 300))
    screen.blit(msg3, (WIDTH // 2 - msg3.get_width() // 2, 325))

    draw_back_button()


def handle_pose_mousedown(pos):
    if pygame.Rect(20, 20, 100, 36).collidepoint(pos):
        return "back"
    return None


# ---- Main loop -----------------------------------------------------------

running = True
while running:
    clock.tick(60)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if SCREEN == "home":
                handle_home_click(event.pos)
            elif SCREEN == "menu":
                handle_menu_click(event.pos)
            elif SCREEN == "edit":
                if handle_edit_mousedown(event.pos) == "back":
                    SCREEN = "menu"
            elif SCREEN == "replay":
                if handle_replay_mousedown(event.pos) == "back":
                    SCREEN = "menu"
            elif SCREEN == "pose":
                if handle_pose_mousedown(event.pos) == "back":
                    SCREEN = "menu"

        elif event.type == pygame.MOUSEBUTTONUP:
            dragging = None

        elif event.type == pygame.MOUSEMOTION and dragging and SCREEN == "edit":
            mx, my = event.pos
            dragging["x"] = max(30, min(WIDTH - 30, mx - drag_offset[0]))
            dragging["y"] = max(60 + 30, min(CANVAS_HEIGHT - 30, my - drag_offset[1]))

    if SCREEN == "home":
        draw_home()
    elif SCREEN == "menu":
        draw_menu()
    elif SCREEN == "edit":
        draw_edit()
    elif SCREEN == "replay":
        draw_replay()
    elif SCREEN == "pose":
        draw_pose_stub()

    pygame.display.flip()

pygame.quit()
sys.exit()