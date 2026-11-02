#!/usr/bin/env python3
# 바닥 주행 시험 — 앞 · 뒤 · 옆으로 왼쪽/오른쪽 · 제자리 회전을 차례로 한다.
# 실행 전: 라즈봇을 바닥에 내려놓고 사방 1m 를 비운다. 시작 전에 삐 소리가 세 번 난다.
# 멈추고 싶으면 Ctrl+C (바퀴는 항상 멈추고 끝난다).
import time

from Raspbot_Lib import Raspbot

# 바퀴 부호 (M1 왼쪽 앞, M2 왼쪽 뒤, M3 오른쪽 앞, M4 오른쪽 뒤)
# Yahboom 메카넘휠 원리 문서의 표와 같다.
STEPS = [
    ("앞으로", (1, 1, 1, 1)),
    ("뒤로", (-1, -1, -1, -1)),
    ("옆으로 왼쪽", (-1, 1, 1, -1)),
    ("옆으로 오른쪽", (1, -1, -1, 1)),
    ("제자리 왼쪽 회전", (-1, -1, 1, 1)),
    ("제자리 오른쪽 회전", (1, 1, -1, -1)),
]
SPEED = 80  # 0~255
SECONDS = 0.8

bot = Raspbot()


def drive(signs, speed):
    for motor_id, sign in enumerate(signs):
        bot.Ctrl_Muto(motor_id, sign * speed)


def stop():
    for motor_id in range(4):
        bot.Ctrl_Muto(motor_id, 0)


try:
    for _ in range(3):
        bot.Ctrl_BEEP_Switch(1)
        time.sleep(0.1)
        bot.Ctrl_BEEP_Switch(0)
        time.sleep(0.9)

    for name, signs in STEPS:
        print(f"-> {name}")
        drive(signs, SPEED)
        time.sleep(SECONDS)
        stop()
        time.sleep(0.6)

    print("[OK] 끝. 여섯 동작이 이름대로 움직였으면 정상입니다.")
    print("     옆으로 갈 때 비스듬히 가는 건 바닥·바퀴 차이로 흔한 일입니다(메카넘휠 문서 참고).")
finally:
    stop()
    del bot
