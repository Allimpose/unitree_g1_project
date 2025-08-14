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


def detect_blue_ball_with_depth(color_frame, depth_frame):
    color_image = np.asanyarray(color_frame.get_data())
    hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 150, 70])
    upper_blue = np.array([140, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(c)
        if radius > 10:
            cx, cy = int(x), int(y)
            z = depth_frame.get_distance(cx, cy)
            if z > 0.1:
                return (cx, cy, z), color_image
    return None, color_image


class Custom:
    def __init__(self):
        self.control_dt_ = 0.02
        self.kp = 60.0
        self.kd = 1.5
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = None
        self.first_update_low_state = False
        self.crc = CRC()

        self.target_right_arm = {
            G1JointIndex.RightShoulderPitch: 0.0,
            G1JointIndex.RightShoulderRoll:  0.0,
            G1JointIndex.RightShoulderYaw:   0.0,
            G1JointIndex.RightElbow:         -0.3,
            G1JointIndex.RightWristRoll:     0.0,
            G1JointIndex.RightWristPitch:    0.0,
            G1JointIndex.RightWristYaw:      0.0
        }

        self.idle_pose = {
            G1JointIndex.RightShoulderPitch: 0.0,
            G1JointIndex.RightShoulderRoll:  -0.2,
            G1JointIndex.RightShoulderYaw:   0.0,
            G1JointIndex.RightElbow:         1.5,
            G1JointIndex.RightWristRoll:     0.0,
            G1JointIndex.RightWristPitch:    0.0,
            G1JointIndex.RightWristYaw:      0.0
        }

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.pipeline.start(self.config)
        print("[INFO] RealSense Camera Started.")

    def Init(self):
        self.arm_sdk_publisher = ChannelPublisher("rt/arm_sdk", LowCmd_)
        self.arm_sdk_publisher.Init()

        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.LowStateHandler, 10)

    def Start(self):
        while not self.first_update_low_state:
            time.sleep(0.1)

        self.move_to_idle_pose(duration=2.0)

        self.lowCmdWriteThreadPtr = RecurrentThread(
            interval=self.control_dt_, target=self.LowCmdWrite, name="control"
        )
        self.lowCmdWriteThreadPtr.Start()

    def LowStateHandler(self, msg: LowState_):
        self.low_state = msg
        if not self.first_update_low_state:
            self.first_update_low_state = True

    def move_to_idle_pose(self, duration=2.0):
        print("[INFO] Moving to idle pose...")
        rate = self.control_dt_
        steps = int(duration / rate)

        for _ in range(steps):
            if self.low_state is None:
                time.sleep(rate)
                continue

            for joint in self.target_right_arm:
                current = self.low_state.motor_state[joint].q
                target = self.idle_pose[joint]
                self.target_right_arm[joint] = (1 - 0.05) * current + 0.05 * target

            self.send_arm_command()
            time.sleep(rate)

        print("[INFO] Idle pose reached.")

    def update_target_by_ball(self, ball_pos):
        if ball_pos is None:
            for joint in self.target_right_arm:
                self.target_right_arm[joint] = self._smooth_approach(
                    self.target_right_arm[joint],
                    self.idle_pose[joint],
                    step=0.02
                )
            return

        x, y, z = ball_pos

        yaw_angle = -(x - 320) / 320 * 0.5

        #  거리(z)에 따라 어깨를 뒤로 젖힘
        pitch_angle = np.clip(0.5 - z, -0.8, 0.5)

        #  화면의 세로 위치(y)에 따라 팔꿈치 조절
        # y가 작을수록(위쪽) → 더 많이 접기
        elbow_angle = np.clip(0.6 - ((240 - y) / 240 * 1.2), -0.8, 0.6)

        #  손목도 약간 반응하게 만들기
        vertical_angle = -(240 - y) / 240 * 0.5

        target_pose = {
            G1JointIndex.RightShoulderYaw:    yaw_angle,
            G1JointIndex.RightShoulderPitch:  pitch_angle,
            G1JointIndex.RightElbow:          elbow_angle,
            G1JointIndex.RightWristYaw:       yaw_angle * 0.5,
            G1JointIndex.RightWristPitch:     vertical_angle * 0.3,
            G1JointIndex.RightWristRoll:      0.0
        }

        for joint, target_q in target_pose.items():
            self.target_right_arm[joint] = self._smooth_approach(
                self.target_right_arm[joint], target_q, step=0.02
            )

    def _smooth_approach(self, current, target, step=0.02):
        if abs(current - target) < step:
            return target
        return current + step if target > current else current - step

    def send_arm_command(self):
        self.low_cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q = 1
        for joint, target_q in self.target_right_arm.items():
            self.low_cmd.motor_cmd[joint].tau = 0.0
            self.low_cmd.motor_cmd[joint].q   = target_q
            self.low_cmd.motor_cmd[joint].dq  = 0.0
            self.low_cmd.motor_cmd[joint].kp  = self.kp
            self.low_cmd.motor_cmd[joint].kd  = self.kd

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.arm_sdk_publisher.Write(self.low_cmd)

    def LowCmdWrite(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if color_frame and depth_frame:
            ball_info, color_image = detect_blue_ball_with_depth(color_frame, depth_frame)
            self.update_target_by_ball(ball_info)

            if ball_info:
                x, y, z = ball_info
                cv2.circle(color_image, (x, y), 10, (255, 0, 0), 2)
                cv2.putText(color_image, f"z={z:.2f}m", (x + 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow("Blue Ball Tracking", color_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                sys.exit(0)

        self.send_arm_command()


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
