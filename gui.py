import time
import sys
import numpy as np
import cv2
import pyrealsense2 as rs
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer

class G1JointIndex:
    # 왼쪽 다리 (Left Leg)
    LeftHipPitch = 0
    LeftHipRoll = 1
    LeftHipYaw = 2
    LeftKnee = 3
    LeftAnklePitch = 4
    LeftAnkleRoll = 5

    # 오른쪽 다리 (Right Leg)
    RightHipPitch = 6
    RightHipRoll = 7
    RightHipYaw = 8
    RightKnee = 9
    RightAnklePitch = 10
    RightAnkleRoll = 11

    # Waist (허리)
    WaistYaw = 12
    WaistRoll = 13
    WaistPitch = 14

    # 왼쪽 팔 (Left Arm)
    LeftShoulderPitch = 15
    LeftShoulderRoll = 16
    LeftShoulderYaw = 17
    LeftElbow = 18
    LeftWristRoll = 19
    LeftWristPitch = 20
    LeftWristYaw = 21

    # 오른쪽 팔 (Right Arm)
    RightShoulderPitch = 22
    RightShoulderRoll = 23
    RightShoulderYaw = 24
    RightElbow = 25
    RightWristRoll = 26
    RightWristPitch = 27
    RightWristYaw = 28

    kNotUsedJoint = 29 # NOTE: Weight

class RealSenseViewer(QWidget):
    def __init__(self):
        super().__init__()

        # RealSense 파이프라인 설정
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.pipeline.start(self.config)

        # GUI 레이아웃
        self.setWindowTitle("RealSense RGB + Depth Viewer")
        self.resize(1365, 768)
        
        # 좌측: RGB 및 Depth 영상, 우측: 모터 상태
        layout = QHBoxLayout()  # 수평 레이아웃
        left_layout = QVBoxLayout()  # RGB와 Depth 영상
        right_layout = QVBoxLayout()  # 모터 상태

        # RGB 영상, Depth 영상 표시
        self.rgb_label = QLabel("RGB 영상")
        self.depth_label = QLabel("Depth 영상")
        left_layout.addWidget(self.rgb_label)
        left_layout.addWidget(self.depth_label)

        # 모터 상태 테이블 표시
        self.motor_status_table = QTableWidget()
        self.motor_status_table.setColumnCount(4)  # 모터 이름, 위치, 속도, 토크
        self.motor_status_table.setHorizontalHeaderLabels(["Motor Name", "Position", "Velocity", "Torque"])
        right_layout.addWidget(self.motor_status_table)

        layout.addLayout(left_layout)
        layout.addLayout(right_layout)
        self.setLayout(layout)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frames)
        self.timer.start(33)

    def update_frames(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            return

        # Color Frame 처리
        color_image = np.asanyarray(color_frame.get_data())

        # Depth Frame 처리
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03),
            cv2.COLORMAP_JET
        )

        rgb_qimg = QImage(color_image.data, color_image.shape[1], color_image.shape[0],
                          color_image.strides[0], QImage.Format_BGR888)
        depth_qimg = QImage(depth_colormap.data, depth_colormap.shape[1], depth_colormap.shape[0],
                            depth_colormap.strides[0], QImage.Format_BGR888)

        self.rgb_label.setPixmap(QPixmap.fromImage(rgb_qimg))
        self.depth_label.setPixmap(QPixmap.fromImage(depth_qimg))

    def update_motor_status(self, motor_states):
        self.motor_status_table.setRowCount(len(motor_states))

        motor_names = {
            0: "LeftHipPitch", 1: "LeftHipRoll", 2: "LeftHipYaw", 3: "LeftKnee", 4: "LeftAnklePitch", 5: "LeftAnkleRoll",
            6: "RightHipPitch", 7: "RightHipRoll", 8: "RightHipYaw", 9: "RightKnee", 10: "RightAnklePitch", 11: "RightAnkleRoll",
            12: "WaistYaw", 13: "WaistRoll", 14: "WaistPitch",
            15: "LeftShoulderPitch", 16: "LeftShoulderRoll", 17: "LeftShoulderYaw", 18: "LeftElbow", 19: "LeftWristRoll", 20: "LeftWristPitch", 21: "LeftWristYaw",
            22: "RightShoulderPitch", 23: "RightShoulderRoll", 24: "RightShoulderYaw", 25: "RightElbow", 26: "RightWristRoll", 27: "RightWristPitch", 28: "RightWristYaw",
            29: "kNotUsedJoint"
        }

        for i, motor in enumerate(motor_states):
            motor_name = motor_names.get(i, f"Motor {i}")
            position = motor['position']
            velocity = motor['velocity']
            torque = motor['torque']

            self.motor_status_table.setItem(i, 0, QTableWidgetItem(motor_name))
            self.motor_status_table.setItem(i, 1, QTableWidgetItem(f"{position:.3f}"))
            self.motor_status_table.setItem(i, 2, QTableWidgetItem(f"{velocity:.3f}"))
            self.motor_status_table.setItem(i, 3, QTableWidgetItem(f"{torque:.3f}"))

    def closeEvent(self, event):
        self.pipeline.stop()
        event.accept()

class Custom:
    def __init__(self, viewer):
        self.control_dt_ = 0.02
        self.kp = 60.0
        self.kd = 1.5
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = None
        self.first_update_low_state = False
        self.crc = CRC()
        self.viewer = viewer
        self.last_update_time = time.time()

    def Init(self):
        self.arm_sdk_publisher = ChannelPublisher("rt/arm_sdk", LowCmd_)
        self.arm_sdk_publisher.Init()

        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.LowStateHandler, 10)

    def Start(self):
        while not self.first_update_low_state:
            time.sleep(1)

    def LowStateHandler(self, msg: LowState_):
        current_time = time.time()
        if current_time - self.last_update_time >= 0.5:
            self.last_update_time = current_time

            self.low_state = msg
            if not self.first_update_low_state:
                self.first_update_low_state = True

            motor_states = []
            for i in range(min(len(msg.motor_state),29)):
                motor = msg.motor_state[i]
                motor_index = i
                position = motor.q  # 위치 (각도)
                velocity = motor.dq  # 속도
                torque = motor.tau_est  # 추정된 토크

                motor_states.append({
                    'position': position,
                    'velocity': velocity,
                    'torque': torque
                })

            self.viewer.update_motor_status(motor_states)

    def LowCmdWrite(self):
        pass

if __name__ == "__main__":
    print("WARNING: Ensure no obstacles around the robot.")
    input("Press Enter to continue...")

    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    app = QApplication(sys.argv)
    viewer = RealSenseViewer()
    viewer.show()

    custom = Custom(viewer)
    custom.Init()
    custom.Start()

    sys.exit(app.exec_())
