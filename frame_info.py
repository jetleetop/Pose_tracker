import random

WINDOW_SIZE = 8
TOTAL_FRAMES = 15

def send_frame(frame_id):
    return random.random() > 0.1

def go_back_n():
    base = 0
    next_seq = 0

    while base < TOTAL_FRAMES:
        while next_seq < base + WINDOW_SIZE and next_seq < TOTAL_FRAMES:
            print(f"송신: 프레임 {next_seq}")
            next_seq += 1

        for i in range(base, next_seq):
            success = send_frame(i)
            if success:
                print(f"  수신: 프레임 {i} → ACK {i+1}")
                base += 1
            else:
                print(f"  수신 실패: 프레임 {i} 손실 → 재전송 준비")
                next_seq = base
                break

go_back_n()
