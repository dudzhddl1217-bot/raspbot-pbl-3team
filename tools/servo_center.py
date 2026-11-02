#!/usr/bin/env python3
# 카메라 짐벌 서보 2개를 '정면 자세'로 맞춘다.
# 공식 이미지는 부팅할 때 이 일을 자동으로 하지만, 우리는 깨끗한 OS 를 깔았으므로 직접 한다.
# 각도는 Yahboom 예제의 초기 자세(Car_Reset) 값: 가로 90도, 세로 25도.
#
# 카메라가 정면을 보지 않으면(제조사 PDF「카메라-짐벌-정렬」 순서):
#   1) 전원을 끄고 짐벌에서 서보 2개를 뺀다(선은 꽂은 채로).
#   2) 전원을 켜고 이 스크립트를 실행한다 -> 서보가 정렬 각도로 돌아가 버틴다.
#   3) 서보 축을 손으로 돌리지 말고, 세로(위) 서보 먼저, 가로(아래) 서보 나중에 끼운다.
import time

from Raspbot_Lib import Raspbot

PAN = 90  # 1번 서보: 가로(좌우) 0~180
TILT = 25  # 2번 서보: 세로(상하) 0~110

bot = Raspbot()
bot.Ctrl_Servo(1, PAN)
time.sleep(0.5)
bot.Ctrl_Servo(2, TILT)
time.sleep(0.5)
print(f"[OK] 서보 정렬: 가로 {PAN}도, 세로 {TILT}도")
del bot
