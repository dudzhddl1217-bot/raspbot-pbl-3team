#!/usr/bin/env python3
# 센서 점검 — 부저 · 불빛 · 초음파 거리 · 라인센서 · 카메라 사진 1장을 차례로 확인한다.
# 사용법은 Yahboom 드라이버(Raspbot_Lib 0.0.2) 소스의 주석 예제를 따른다.
import os
import time

from Raspbot_Lib import Raspbot

bot = Raspbot()


def read_byte(reg):
    data = bot.read_data_array(reg, 1)  # 통신 오류면 None (드라이버가 'I2C error' 를 찍는다)
    return None if data is None else data[0]


def check_buzzer_and_light():
    print("[1/4] 부저와 불빛: 삐 소리 → 빨강 → 초록 → 파랑 → 노랑 → 꺼짐")
    bot.Ctrl_BEEP_Switch(1)
    time.sleep(0.15)
    bot.Ctrl_BEEP_Switch(0)
    for color in range(4):  # 0 빨강 · 1 초록 · 2 파랑 · 3 노랑
        bot.Ctrl_WQ2812_ALL(1, color)
        time.sleep(0.5)
    bot.Ctrl_WQ2812_ALL(0, 0)


def check_ultrasonic(times=6):
    print("[2/4] 초음파 거리: 라즈봇 앞에 손을 가까이 댔다 멀리 했다 해 보세요")
    bot.Ctrl_Ulatist_Switch(1)
    time.sleep(1)
    for _ in range(times):
        high, low = read_byte(0x1b), read_byte(0x1a)
        if high is None or low is None:
            print("   읽기 실패")
        else:
            print(f"   거리 {high << 8 | low} mm")
        time.sleep(0.5)
    bot.Ctrl_Ulatist_Switch(0)


def check_line_sensor(seconds=10):
    print(f"[3/4] 라인센서 ({seconds}초): 검은 테이프를 센서 4개 아래로 지나가게 해 보세요")
    print("      값이 바뀌고 센서 위 작은 불이 켜졌다 꺼지면 정상입니다")
    end, last = time.monotonic() + seconds, None
    while time.monotonic() < end:
        track = read_byte(0x0a)
        if track is not None:
            bits = ((track >> 3) & 1, (track >> 2) & 1, (track >> 1) & 1, track & 1)
            if bits != last:
                print("   x1 x2 x3 x4 = {}  {}  {}  {}".format(*bits))
                last = bits
        time.sleep(0.1)


RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def check_camera(path=os.path.join(RESULTS, "camera_test.jpg")):
    print("[4/4] 카메라: 사진 1장을 찍어 저장합니다")
    try:
        import cv2
    except ImportError:
        print("   OpenCV 가 없습니다 → sudo apt update && sudo apt install -y python3-opencv 후 다시 실행")
        return
    path = os.path.expanduser(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for index in (0, 1):  # 보통 0번이 USB 카메라
        cap = cv2.VideoCapture(index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        ok, frame = False, None
        for _ in range(10):  # 켜자마자 몇 장은 어둡게 나와서 버린다
            ok, frame = cap.read()
        cap.release()
        if ok:
            cv2.imwrite(path, frame)
            print(f"   저장: {path} ({frame.shape[1]}x{frame.shape[0]}, 카메라 {index}번)")
            return
    print("   사진을 못 찍었습니다 → 카메라 USB 선을 확인하세요 (ls /dev/video* 로 장치 확인)")


try:
    check_buzzer_and_light()
    check_ultrasonic()
    check_line_sensor()
    check_camera()
    print("[OK] 센서 점검 끝")
finally:
    bot.Ctrl_WQ2812_ALL(0, 0)
    bot.Ctrl_Ulatist_Switch(0)
    del bot
