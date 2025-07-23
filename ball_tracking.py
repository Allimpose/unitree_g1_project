import time
import sys
import cv2
import numpy as np
import pyrealsense2 as rs

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread


class G1JointIndex:
    RightShoulderPitch = 22
    RightShoulderRoll  = 23
    RightShoulderYaw   = 24
    RightElbow         = 25
    RightWristRoll     = 26
    RightWristPitch    = 27
    RightWristYaw      = 28
    kNotUsedJoint      = 29


def detect_blue_ball(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 150, 70])
    upper_blue = np.array([140, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(c)
        if radius > 10:
            return (int(x), int(y))
    return None


class Custom:
    def __init__(self):
        self.control_dt_ = 0.02
        self.kp = 60.0
        self.kd = 1.5
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = None
        self.first_update_low_state = False
        self.crc = CRC()

        # 현재 팔 관절 목표값
        self.target_right_arm = {
            G1JointIndex.RightShoulderPitch: 0.0,
            G1JointIndex.RightShoulderRoll:  0.0,
            G1JointIndex.RightShoulderYaw:   0.0,
            G1JointIndex.RightElbow:         -0.3,
            G1JointIndex.RightWristRoll:     0.0,
            G1JointIndex.RightWristPitch:    0.0,
            G1JointIndex.RightWristYaw:      0.0
        }

        # 차렷(기본) 자세
        self.idle_pose = {
            G1JointIndex.RightShoulderPitch: 0.0,
            G1JointIndex.RightShoulderRoll:  -0.2,  # 어깨를 살짝 오른쪽으로
            G1JointIndex.RightShoulderYaw:   0.0,
            G1JointIndex.RightElbow:         1.57079632,  # 90도 접힘
            G1JointIndex.RightWristRoll:     0.0,
            G1JointIndex.RightWristPitch:    0.0,
            G1JointIndex.RightWristYaw:      0.0
        }

        # RealSense 초기화
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(self.config)
        print("[INFO] RealSense Camera Started.")

    def Init(self):
        self.arm_sdk_publisher = ChannelPublisher("rt/arm_sdk", LowCmd_)
        self.arm_sdk_publisher.Init()

        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.LowStateHandler, 10)

    def Start(self):
        self.lowCmdWriteThreadPtr = RecurrentThread(
            interval=self.control_dt_, target=self.LowCmdWrite, name="control"
        )
        while not self.first_update_low_state:
            time.sleep(1)
        self.lowCmdWriteThreadPtr.Start()

    def LowStateHandler(self, msg: LowState_):
        self.low_state = msg
        if not self.first_update_low_state:
            self.first_update_low_state = True

    def update_target_by_ball(self, ball_pos):
        if ball_pos is None:
            # 공이 없으면 차렷 자세로 천천히 이동
            for joint in self.target_right_arm:
                self.target_right_arm[joint] = self._smooth_approach(
                    self.target_right_arm[joint],
                    self.idle_pose[joint],
                    step=0.02
                )
            return

        x, y = ball_pos
        yaw_angle   = -(x - 320) / 320 * 0.5
        pitch_angle = -(240 - y) / 240 * 0.5

        # 목표 관절값 (공 위치 기반)
        target_pose = {
            G1JointIndex.RightShoulderYaw:   yaw_angle,
            G1JointIndex.RightShoulderPitch: pitch_angle,
            G1JointIndex.RightElbow:        -0.3,
            G1JointIndex.RightWristYaw:     yaw_angle * 0.5,
            G1JointIndex.RightWristPitch:   pitch_angle * 0.3,
            G1JointIndex.RightWristRoll:    0.0
        }

        # 현재 관절값에서 목표로 천천히 이동
        for joint, target_q in target_pose.items():
            self.target_right_arm[joint] = self._smooth_approach(
                self.target_right_arm[joint], target_q, step=0.02
            )

    def _smooth_approach(self, current, target, step=0.02):
        """현재값에서 target으로 부드럽게 이동"""
        if abs(current - target) < step:
            return target
        return current + step if target > current else current - step

    def LowCmdWrite(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if color_frame:
            color_image = np.asanyarray(color_frame.get_data())
            ball_pos = detect_blue_ball(color_image)
            self.update_target_by_ball(ball_pos)

            if ball_pos is not None:
                cv2.circle(color_image, ball_pos, 10, (255, 0, 0), 2)
            cv2.imshow("Blue Ball Detection", color_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                sys.exit(0)

        # 팔 명령 전송
        self.low_cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q = 1
        for joint, target_q in self.target_right_arm.items():
            self.low_cmd.motor_cmd[joint].tau = 0.0
            self.low_cmd.motor_cmd[joint].q   = target_q
            self.low_cmd.motor_cmd[joint].dq  = 0.0
            self.low_cmd.motor_cmd[joint].kp  = self.kp
            self.low_cmd.motor_cmd[joint].kd  = self.kd

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.arm_sdk_publisher.Write(self.low_cmd)


if __name__ == "__main__":
    print("WARNING: Ensure no obstacles around the robot.")
    input("Press Enter to continue...")

    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    custom = Custom()
    custom.Init()
    custom.Start()

    while True:
        time.sleep(1)

