import cv2
import mediapipe as mp
import time
import pyrealsense2 as rs
import numpy as np
from pymodbus.client.sync import ModbusTcpClient
import math


mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils


regdict = {
    'angleSet': 1486,  # 손 각도 설정
    'forceSet': 1498,  # 손 힘 설정
    'speedSet': 1522,  # 손 속도 설정
}


def calculate_angle(a, b, c):

    ab = [b.x - a.x, b.y - a.y, b.z - a.z]
    bc = [c.x - b.x, c.y - b.y, c.z - b.z]


    dot_product = ab[0] * bc[0] + ab[1] * bc[1] + ab[2] * bc[2]
    

    ab_magnitude = math.sqrt(ab[0]**2 + ab[1]**2 + ab[2]**2)
    bc_magnitude = math.sqrt(bc[0]**2 + bc[1]**2 + bc[2]**2)
    

    cosine_angle = dot_product / (ab_magnitude * bc_magnitude)
    angle = math.acos(cosine_angle)


    angle_deg = math.degrees(angle)
    return angle_deg

def angle_to_motor_value(angle, sensitivity=1.0):
    # 민감도(sensitivity)를 곱해서 더 빠르게 변화하도록 함
    scaled_angle = angle * sensitivity
    
    motor_value = 1000 - (scaled_angle / 180) * 1000
    motor_value = max(0, min(1000, motor_value))
    return int(motor_value)

def get_finger_angles(landmarks):
    angles = {}
    
    pinky_angle = calculate_angle(landmarks[17], landmarks[18], landmarks[19])
    angles['pinky'] = angle_to_motor_value(pinky_angle)
    
    ring_angle = calculate_angle(landmarks[13], landmarks[14], landmarks[15])
    angles['ring'] = angle_to_motor_value(ring_angle)
    
    middle_angle = calculate_angle(landmarks[9], landmarks[10], landmarks[11])
    angles['middle'] = angle_to_motor_value(middle_angle)
    
    index_angle = calculate_angle(landmarks[5], landmarks[6], landmarks[7])
    angles['index'] = angle_to_motor_value(index_angle)
    
    # 엄지는 민감도 1.5배로 조정
    thumb_angle = calculate_angle(landmarks[1], landmarks[2], landmarks[3])
    angles['thumb'] = angle_to_motor_value(thumb_angle, sensitivity=1.5)
    
    return angles


def open_modbus(ip, port, timeout=10):
    start_time = time.time()
    client = ModbusTcpClient(ip, port)
    
    while (time.time() - start_time) < timeout:
        if client.connect():
            print(f"성공적으로 {ip}:{port}에 연결되었습니다.")
            return client
        else:
            print(f"{ip}:{port}에 연결 실패, 재시도 중...")
            time.sleep(1)
    
    print(f"연결 실패, 시간 초과: {ip}:{port}")
    return None

def write_register(client, address, values):
    client.write_registers(address, values)


previous_angles = None
previous_forces = None
previous_speeds = None


update_interval = 0.1


def track_and_control_robot(ip):
    global previous_angles, previous_forces, previous_speeds

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    client = open_modbus(ip, 6000, timeout=10)
    if client is None:
        return

    last_update_time = time.time()

    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue


        frame = np.asanyarray(color_frame.get_data())


        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)


        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:

                angles = get_finger_angles(landmarks.landmark)
                print(f"각 손가락 모터 제어 값: {angles}")


                current_time = time.time()
                if (current_time - last_update_time) >= update_interval:
                    if previous_angles != angles:

                        write_register(client, regdict['angleSet'], [int(angle) for angle in angles.values()])
                        previous_angles = angles
                        last_update_time = current_time


                mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)


        cv2.imshow('Hand Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    pipeline.stop()
    cv2.destroyAllWindows()
    client.close()
    print(f"{ip} - 연결 종료")

if __name__ == '__main__':
    ip = '192.168.123.210'
    track_and_control_robot(ip)
