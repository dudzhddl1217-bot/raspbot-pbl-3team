#!/usr/bin/env python3
# 노트북 키보드로 라즈봇 조종하기 — VS Code 터미널(또는 SSH 창)에서 실행한다.
#   w / s : 앞 / 뒤          a / d : 옆으로 왼쪽 / 오른쪽
#   q / e : 제자리 회전       스페이스 : 정지
#   + / - : 속도 올리기 / 내리기     x : 끝내기
# 키를 누르고 있는 동안 움직이고, 손을 떼면 잠시 뒤(AUTO_STOP 초) 저절로 멈춘다.
import curses
import time

from Raspbot_Lib import Raspbot

# 바퀴 부호 (M1 왼쪽 앞, M2 왼쪽 뒤, M3 오른쪽 앞, M4 오른쪽 뒤) — drive_test.py 와 같은 표
MOVES = {
    "w": (1, 1, 1, 1),
    "s": (-1, -1, -1, -1),
    "a": (-1, 1, 1, -1),
    "d": (1, -1, -1, 1),
    "q": (-1, -1, 1, 1),
    "e": (1, 1, -1, -1),
}
SPEED_MIN, SPEED_MAX, SPEED_STEP = 40, 150, 20
AUTO_STOP = 0.6  # 키 반복이 시작되기 전 공백(약 0.5초)보다 길게

bot = Raspbot()


def drive(signs, speed):
    for motor_id, sign in enumerate(signs):
        bot.Ctrl_Muto(motor_id, sign * speed)


def stop():
    for motor_id in range(4):
        bot.Ctrl_Muto(motor_id, 0)


def main(screen):
    curses.curs_set(0)
    screen.timeout(50)
    speed, moving, last_press = 80, None, 0.0
    while True:
        screen.erase()
        screen.addstr(0, 0, "Raspbot teleop")
        screen.addstr(1, 0, "w/s fwd/back   a/d side   q/e turn   space stop   +/- speed   x quit")
        screen.addstr(3, 0, f"speed {speed:3d}   move {moving or 'stop'}")
        ch = screen.getch()
        now = time.monotonic()
        if 0 <= ch < 256:
            key = chr(ch).lower()
            if key == "x":
                break
            if key == " ":
                moving = None
                stop()
            elif key in "+=":
                speed = min(SPEED_MAX, speed + SPEED_STEP)
            elif key in "-_":
                speed = max(SPEED_MIN, speed - SPEED_STEP)
            elif key in MOVES:
                moving, last_press = key, now
                drive(MOVES[key], speed)
        if moving and now - last_press > AUTO_STOP:
            moving = None
            stop()


try:
    curses.wrapper(main)
finally:
    stop()
    del bot
    print("[OK] 조종 끝 — 바퀴 정지")
