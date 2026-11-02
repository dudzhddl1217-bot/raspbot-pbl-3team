#!/usr/bin/env python3
# 라즈봇 V2 동작 확인 — 삐 소리 한 번, 그다음 바퀴 4개를 하나씩 1초씩 앞으로 돌린다.
# 실행 전에 라즈봇을 상자 위에 올려 바퀴가 바닥에 닿지 않게 한다(책상에서 달려 나가지 않도록).
# 모터 번호는 Yahboom 예제 코드 기준 0~3 (M1 왼쪽 앞 · M2 왼쪽 뒤 · M3 오른쪽 앞 · M4 오른쪽 뒤).
import time

from Raspbot_Lib import Raspbot

WHEELS = ["M1 왼쪽 앞", "M2 왼쪽 뒤", "M3 오른쪽 앞", "M4 오른쪽 뒤"]
SPEED = 80  # -255(뒤) ~ 255(앞)

bot = Raspbot()
try:
    bot.Ctrl_BEEP_Switch(1)
    time.sleep(0.2)
    bot.Ctrl_BEEP_Switch(0)

    for motor_id, name in enumerate(WHEELS):
        print(f"[{motor_id}] {name} -> 1초 회전")
        bot.Ctrl_Muto(motor_id, SPEED)
        time.sleep(1)
        bot.Ctrl_Muto(motor_id, 0)
        time.sleep(0.3)

    print("[OK] 끝. 네 바퀴가 차례로 '앞쪽'으로 돌았으면 정상입니다.")
    print("     어느 바퀴가 안 돌거나 거꾸로 돌면 그 모터 선(M1~M4)을 확인하세요.")
finally:
    for motor_id in range(4):
        bot.Ctrl_Muto(motor_id, 0)
    del bot
