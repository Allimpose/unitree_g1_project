#!/usr/bin/env python3
import argparse
import time

from unitree_sdk2_python.unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
from unitree_sdk2_python.unitree_sdk2py.core.channel import ChannelFactory


def str2floats(s: str):
    return [float(x) for x in s.split()]


def main():
    parser = argparse.ArgumentParser(description="Unitree G1 Python LocoClient")

    # 네트워크 인터페이스
    parser.add_argument("--network_interface", default="lo",
                        help="network interface (e.g. lo, eth0, wlan0)")

    # 기본 동작
    parser.add_argument("--damp", action="store_true", help="Enter damping mode")
    parser.add_argument("--start", action="store_true", help="Start walking")
    parser.add_argument("--sit", action="store_true", help="Sit down")
    parser.add_argument("--stand_up", action="store_true", help="Stand up")
    parser.add_argument("--zero_torque", action="store_true", help="Zero torque mode")
    parser.add_argument("--stop_move", action="store_true", help="Stop movement")
    parser.add_argument("--high_stand", action="store_true", help="High stand posture")
    parser.add_argument("--low_stand", action="store_true", help="Low stand posture")

    # 모드 및 상태 제어
    parser.add_argument("--balance_stand", type=int, help="Balance stand mode (int)")
    parser.add_argument("--set_fsm_id", type=int, help="Set FSM ID")
    parser.add_argument("--set_balance_mode", type=int, help="Set balance mode")
    parser.add_argument("--set_stand_height", type=float, help="Set stand height")
    parser.add_argument("--set_velocity", type=str,
                        help='"vx vy omega [duration]" e.g. "0.2 0 0 3"')
    parser.add_argument("--move", type=str,
                        help='"vx vy omega" e.g. "0.2 0 0"')

    # 제스처 동작
    parser.add_argument("--wave_hand", action="store_true", help="Wave hand")
    parser.add_argument("--wave_hand_with_turn", action="store_true", help="Wave hand with turn")
    parser.add_argument("--shake_hand", action="store_true", help="Shake hand")
    parser.add_argument("--set_task_id", type=int, help="Set custom task ID")

    args = parser.parse_args()

    # DDS 초기화 (필수!)
    factory = ChannelFactory()
    factory.Init(0, args.network_interface)

    # LocoClient 초기화
    client = LocoClient()
    client.Init()

    # --------------------
    # 동작 실행
    # --------------------
    if args.damp:
        print("[INFO] Entering damp mode...")
        client.Damp()

    if args.start:
        print("[INFO] Start walking...")
        client.Start()

    if args.sit:
        print("[INFO] Sit down...")
        client.Sit()

    if args.stand_up:
        print("[INFO] Stand up...")
        client.Lie2StandUp()

    if args.zero_torque:
        print("[INFO] Zero torque...")
        client.ZeroTorque()

    if args.stop_move:
        print("[INFO] Stop moving...")
        client.StopMove()

    if args.high_stand:
        print("[INFO] High stand posture...")
        client.HighStand()

    if args.low_stand:
        print("[INFO] Low stand posture...")
        client.LowStand()

    if args.balance_stand is not None:
        print(f"[INFO] Balance stand: {args.balance_stand}")
        client.BalanceStand(args.balance_stand)

    if args.set_fsm_id is not None:
        print(f"[INFO] Set FSM ID: {args.set_fsm_id}")
        client.SetFsmId(args.set_fsm_id)

    if args.set_balance_mode is not None:
        print(f"[INFO] Set balance mode: {args.set_balance_mode}")
        client.SetBalanceMode(args.set_balance_mode)

    if args.set_stand_height is not None:
        print(f"[INFO] Set stand height: {args.set_stand_height}")
        client.SetStandHeight(args.set_stand_height)

    if args.set_velocity:
        values = str2floats(args.set_velocity)
        if len(values) == 3:
            vx, vy, omega = values
            duration = 1.0
        elif len(values) == 4:
            vx, vy, omega, duration = values
        else:
            raise ValueError("Invalid velocity format")
        print(f"[INFO] Set velocity: {values}")
        client.SetVelocity(vx, vy, omega, duration)

    if args.move:
        vx, vy, omega = str2floats(args.move)
        print(f"[INFO] Move with velocity: {vx}, {vy}, {omega}")
        client.Move(vx, vy, omega)

    if args.wave_hand:
        print("[INFO] Wave hand")
        client.WaveHand()

    if args.wave_hand_with_turn:
        print("[INFO] Wave hand with turn")
        client.WaveHand(turn_flag=True)

    if args.shake_hand:
        print("[INFO] Shake hand start (10s)...")
        client.ShakeHand(0)
        time.sleep(10)
        client.ShakeHand(1)
        print("[INFO] Shake hand end")

    if args.set_task_id is not None:
        print(f"[INFO] Set task ID: {args.set_task_id}")
        client.SetTaskId(args.set_task_id)


if __name__ == "__main__":
    main()
