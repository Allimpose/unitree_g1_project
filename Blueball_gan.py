
from __future__ import annotations

import argparse
import math
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np

# --------- RealSense detector ----------
try:
    import cv2
    import pyrealsense2 as rs
except Exception:
    cv2 = None
    rs = None

@dataclass
class DetectorConfig:
    pitch_deg: float = 42.4
    cam_height_m: float = 1.25
    flip_lr: bool = False
    ball_radius_m: float = 0.025
    lower_blue: Tuple[int,int,int] = (95,120,60)
    upper_blue: Tuple[int,int,int] = (130,255,255)
    blur_ksize: int = 5
    min_area: int = 300
    depth_win: int = 5
    smooth_win: int = 5
    use_multipoint: bool = True
    multi_radius: int = 7
    sample_max: int = 200
    show: bool = False
    timeout_s: float = 15.0
    show_hold_ms: int = 800
    show_keep: bool = True      # 검출 후 q/ESC까지 창 유지

# 하드 제한
X_MAX_M = 1.5   # x는 절대 1.5m 초과 금지
Z_MIN_M = 0.0   # z는 절대 음수 금지

class RealSenseBlueBallDetector:
    def __init__(self, cfg: DetectorConfig):
        if cv2 is None or rs is None:
            raise RuntimeError("OpenCV/RealSense 필요(cv2, pyrealsense2).")
        self.cfg = cfg
        self.pipeline: Optional[rs.pipeline] = None
        self.align = None
        self.depth_scale = 0.001
        self.intr = None
        self.hist = deque(maxlen=cfg.smooth_win)
        self.theta = math.radians(cfg.pitch_deg)
        self._kernel = np.ones((5,5), np.uint8)

    def start(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640,480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640,480, rs.format.bgr8, 30)
        self.align = rs.align(rs.stream.color)
        prof = self.pipeline.start(config)
        depth_sensor = prof.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        color_stream = prof.get_stream(rs.stream.color).as_video_stream_profile()
        self.intr = color_stream.get_intrinsics()
        self._spatial = rs.spatial_filter()
        self._temporal = rs.temporal_filter()
        self._hole = rs.hole_filling_filter()
        if self.cfg.show:
            cv2.namedWindow("Color", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Mask",  cv2.WINDOW_NORMAL)

    def stop(self):
        try:
            if self.pipeline is not None:
                self.pipeline.stop()
        finally:
            if self.cfg.show and cv2 is not None:
                try: cv2.destroyAllWindows()
                except: pass

    def _apply_filters(self, depth_frame: rs.depth_frame) -> rs.depth_frame:
        f = depth_frame
        f = self._spatial.process(f)
        f = self._temporal.process(f)
        f = self._hole.process(f)
        return f

    @staticmethod
    def _median_depth(depth_m: np.ndarray, u: float, v: float, win: int) -> float:
        H, W = depth_m.shape
        u0 = max(0, int(u) - win//2); v0 = max(0, int(v) - win//2)
        u1 = min(W, u0 + win);        v1 = min(H, v0 + win)
        patch = depth_m[v0:v1, u0:u1].astype(np.float32)
        patch = patch[patch > 0]
        if patch.size == 0: return 0.0
        return float(np.median(patch))

    def _deproject_multipoint(self, mask: np.ndarray, depth_m: np.ndarray, cx: float, cy: float):
        Hm, Wm = mask.shape; Hd, Wd = depth_m.shape
        R = self.cfg.multi_radius
        y0 = max(0, int(cy)-R); y1 = min(Hm, int(cy)+R+1)
        x0 = max(0, int(cx)-R); x1 = min(Wm, int(cx)+R+1)
        sub = mask[y0:y1, x0:x1] > 0
        ys, xs = np.where(sub)
        if ys.size == 0: return [],[],[]
        xs = (xs + x0).astype(np.int32); ys = (ys + y0).astype(np.int32)
        if xs.size > self.cfg.sample_max:
            step = max(1, xs.size//self.cfg.sample_max)
            xs = xs[::step]; ys = ys[::step]
        ptsX, ptsY, ptsZ = [],[],[]
        for u,v in zip(xs,ys):
            if v<0 or v>=Hd or u<0 or u>=Wd:
                continue
            d = float(depth_m[v,u])
            if d<=0.0:
                continue
            Xc,Yc,Zc = rs.rs2_deproject_pixel_to_point(self.intr, [float(u),float(v)], d)
            ptsX.append(Xc); ptsY.append(Yc); ptsZ.append(Zc)
        return ptsX,ptsY,ptsZ

    def get_xyz_once(self) -> Tuple[float,float,float]:
        import time
        t0 = time.time()
        while True:
            if (time.time()-t0) > self.cfg.timeout_s:
                raise TimeoutError("파란 공 감지 타임아웃")
            frames = self.pipeline.wait_for_frames()
            aligned = self.align.process(frames)
            depth_raw = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not depth_raw or not color_frame: 
                continue
            depth_f = self._apply_filters(depth_raw)
            depth_m = np.asanyarray(depth_f.get_data()).astype(np.float32) * self.depth_scale
            color = np.asanyarray(color_frame.get_data())

            # Show even when not detected (창이 유지되도록)
            def _show_now():
                if self.cfg.show:
                    cv2.imshow("Color", color); cv2.imshow("Mask", mask)
                    cv2.waitKey(1)

            blurred = cv2.GaussianBlur(color, (self.cfg.blur_ksize,self.cfg.blur_ksize), 0)
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
            lower = np.array(self.cfg.lower_blue, np.uint8)
            upper = np.array(self.cfg.upper_blue, np.uint8)
            mask = cv2.inRange(hsv, lower, upper)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel, iterations=2)

            cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                _show_now()
                continue
            cnt = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(cnt) < self.cfg.min_area:
                _show_now()
                continue

            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = M["m10"]/M["m00"]; cy = M["m01"]/M["m00"]
            else:
                (cx,cy),_ = cv2.minEnclosingCircle(cnt)

            ptsX,ptsY,ptsZ = [],[],[]
            if self.cfg.use_multipoint:
                ptsX,ptsY,ptsZ = self._deproject_multipoint(mask, depth_m, cx, cy)
            if (not self.cfg.use_multipoint) or len(ptsZ)==0:
                dep = self._median_depth(depth_m, cx, cy, self.cfg.depth_win)
                if dep>0.0:
                    Xc,Yc,Zc = rs.rs2_deproject_pixel_to_point(self.intr, [float(cx),float(cy)], dep)
                    ptsX=[Xc]; ptsY=[Yc]; ptsZ=[Zc]
            if len(ptsZ)==0:
                _show_now()
                continue

            # 중앙값 (카메라 프레임)
            Xc = float(np.median(ptsX)); Yc = float(np.median(ptsY)); Zc = float(np.median(ptsZ))
            # 피치 보정
            xr = Xc; yr = -Yc; zr = Zc
            y_corr = yr*math.cos(self.theta) - zr*math.sin(self.theta)
            x_corr = xr

            # 좌표 계산
            x = Zc + self.cfg.ball_radius_m   # 전방 깊이 + 반지름(표면→중심)
            y_nominal = -x_corr               # 현장 기준에 맞춘 좌우 부호(필요시 flip)
            y = (-y_nominal) if self.cfg.flip_lr else y_nominal
            z = self.cfg.cam_height_m + y_corr

            # 하드 클램프 (요청사항)
            x = min(max(0.0, x), X_MAX_M)     # 0 ≤ x ≤ 1.5
            z = max(Z_MIN_M, z)               # z ≥ 0

            # 이동 평균
            self.hist.append((x,y,z))
            sx = float(np.mean([p[0] for p in self.hist]))
            sy = float(np.mean([p[1] for p in self.hist]))
            sz = float(np.mean([p[2] for p in self.hist]))

            if self.cfg.show:
                # 디스플레이용 bbox/텍스트
                bx, by, bw, bh = cv2.boundingRect(cnt)
                cv2.circle(color, (int(cx), int(cy)), 6, (0, 255, 255), -1)
                cv2.rectangle(color, (int(bx), int(by)),
                              (int(bx + bw), int(by + bh)), (255, 0, 0), 2)
                txt = f"x={sx:.3f} y={sy:.3f} z={sz:.3f} m (clamped)"
                cv2.putText(color, txt, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0),3)
                cv2.putText(color, txt, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255),2)
                cv2.imshow("Color", color); cv2.imshow("Mask", mask)
                if self.cfg.show_keep:
                    # q/ESC가 눌릴 때까지 유지
                    while True:
                        key = cv2.waitKey(30) & 0xFF
                        if key in (27, ord('q')):
                            break
                else:
                    cv2.waitKey(max(1, int(self.cfg.show_hold_ms)))

            return sx,sy,sz

# --------- G1 runner ----------
def load_policy(path):
    from stable_baselines3 import PPO  # type: ignore
    print(f"[policy] Loading PPO model from {path} …")
    policy = PPO.load(str(path), device="cpu")
    policy.set_parameters(policy.get_parameters())
    policy.policy.set_training_mode(False)
    return policy

def make_env_headless(right_arm: bool):
    import sys
    sys.path.append("/home/unitree/g1_project/RL-shenanigans")
    import g1_arm_rl_env as _env
    return _env.G1ArmReachEnv(render_mode="none", right_arm=right_arm)

class RobotBridge:
    def __init__(self, iface: str, domain: int):
        try:
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
            from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        except Exception:
            print("[robot] SDK-2 not present – robot output disabled")
            self.ok = False; return
        try:
            ChannelFactoryInitialize(domain, iface)
            self._pub = ChannelPublisher("rt/arm_sdk", LowCmd_); self._pub.Init()
            self._cmd = unitree_hg_msg_dds__LowCmd_()
            for mc in self._cmd.motor_cmd:
                mc.mode = 0; mc.kp = 40.0; mc.kd = 1.0
            if 29 < len(self._cmd.motor_cmd): self._cmd.motor_cmd[29].q = 1.0
            try:
                from unitree_sdk2py.utils.crc import CRC
                self._crc = CRC()
            except Exception: self._crc = None
            self.ok = True
        except Exception as e:
            print(f"[robot] DDS init failed – disabled ({e})"); self.ok = False

    def send_qpos(self, q: Dict[int,float]) -> None:
        if not self.ok: return
        for idx,val in q.items():
            if idx>=29: continue
            if idx < len(self._cmd.motor_cmd): self._cmd.motor_cmd[idx].q = float(val)
        if self._crc is not None:
            if hasattr(self._crc,"Crc"): self._cmd.crc = self._crc.Crc(self._cmd)
            elif hasattr(self._crc,"calculate_crc"): self._cmd.crc = self._crc.calculate_crc(self._cmd)
        self._pub.Write(self._cmd)

JOINTS: List[Tuple[int,str,str]] = [
    (15,"L shoulder-pitch","left_shoulder_pitch"),
    (16,"L shoulder-roll","left_shoulder_roll"),
    (17,"L shoulder-yaw","left_shoulder_yaw"),
    (18,"L elbow","left_elbow"),
    (19,"L wrist-roll","left_wrist_roll"),
    (20,"L wrist-pitch","left_wrist_pitch"),
    (21,"L wrist-yaw","left_wrist_yaw"),
    (22,"R shoulder-pitch","right_shoulder_pitch"),
    (23,"R shoulder-roll","right_shoulder_roll"),
    (24,"R shoulder-yaw","right_shoulder_yaw"),
    (25,"R elbow","right_elbow"),
    (26,"R wrist-roll","right_wrist_roll"),
    (27,"R wrist-pitch","right_wrist_pitch"),
    (28,"R wrist-yaw","right_wrist_yaw"),
]

def apply_initial_pose_if_available(env):
    try:
        from g1_initial_pose import POSE_DICT  # type: ignore
        import mujoco as _mj
        def _joint_name(idx:int)->str|None:
            left = [
                "left_shoulder_pitch_joint","left_shoulder_roll_joint","left_shoulder_yaw_joint",
                "left_elbow_joint","left_wrist_roll_joint","left_wrist_pitch_joint","left_wrist_yaw_joint",
            ]
            right = [
                "right_shoulder_pitch_joint","right_shoulder_roll_joint","right_shoulder_yaw_joint",
                "right_elbow_joint","right_wrist_roll_joint","right_wrist_pitch_joint","right_wrist_yaw_joint",
            ]
            if idx==12: return "waist_yaw_joint"
            if 15<=idx<=21: return left[idx-15]
            if 22<=idx<=28: return right[idx-22]
            return None
        for midx,q in POSE_DICT.items():
            jname = _joint_name(midx); 
            if jname is None: continue
            jid = _mj.mj_name2id(env.model, _mj.mjtObj.mjOBJ_JOINT, jname)
            if jid==-1: continue
            qadr = int(env.model.jnt_qposadr[jid]); env.data.qpos[qadr] = float(q)
        _mj.mj_forward(env.model, env.data)
        if hasattr(env,"_fk"):
            p_hand = env._fk(); env.p_goal[:] = p_hand
            if env._goal_mid != -1: env.data.mocap_pos[env._goal_mid] = env.p_goal
        try:
            if hasattr(env, "_park_qadr") and hasattr(env, "_park_rest_q"):
                new_rest = [float(env.data.qpos[qadr]) for qadr in env._park_qadr]
                env._park_rest_q = np.array(new_rest, dtype=np.float32)
        except: pass
    except: pass

def run_goto(goal: np.ndarray, model_path: str, iface: str, domain: int, right_arm: bool,
             rate: float, reach_eps: float, timeout: float, action_mult: float,
             no_prompt: bool, approach_speed: float) -> None:
    from types import SimpleNamespace
    import time
    import mujoco as _mj

    print(f"[goto] Goal set to x={goal[0]:.3f} y={goal[1]:.3f} z={goal[2]:.3f} (m)")
    policy = load_policy(PathLike(model_path))
    env = make_env_headless(right_arm=right_arm)

    obs, _ = env.reset()
    apply_initial_pose_if_available(env)

    # 안전 워크스페이스 박스
    goal = np.clip(goal, [-0.1,-0.6,0.4], [0.6,0.6,1.4])

    # 슬로우 어프로치: 현재 손 위치에서 시작하는 중간 목표 g_cur
    if hasattr(env,"_fk"): g_cur = env._fk().copy()
    else: g_cur = goal.copy()
    env.p_goal[:] = g_cur
    if env._goal_mid != -1: env.data.mocap_pos[env._goal_mid] = env.p_goal

    robot = RobotBridge(iface, domain)
    if not robot.ok:
        print("[warn] Robot DDS not available. (Check iface/domain/SDK-2)")

    motor_qadr: Dict[int,int] = {}
    for idx,_,mj_short in JOINTS:
        jname_joint = mj_short + "_joint"
        jid = _mj.mj_name2id(env.model, _mj.mjtObj.mjOBJ_JOINT, jname_joint)
        if jid != -1:
            motor_qadr[idx] = int(env.model.jnt_qposadr[jid])

    if not no_prompt:
        input("⚠️  안전 확인(E-Stop 준비 등). Enter로 시작… ")

    speed = SimpleNamespace(dt=max(0.005, rate), mult=float(action_mult))
    start_t = time.time(); last_robot_send = 0.0
    last_safe_qpos = None; hold_mode = False; collision_freeze = False
    last_status = 0.0

    print("[goto] Running… (Ctrl+C to abort)")
    try:
        while True:
            # (1) 중간 목표를 최종 목표쪽으로 approach_speed*dt 만큼 이동
            vec = goal - g_cur
            dist_goal = float(np.linalg.norm(vec))
            if dist_goal > 1e-6:
                step = min(approach_speed * speed.dt, dist_goal)
                g_cur = np.clip(g_cur + (vec/dist_goal) * step, [-0.1,-0.6,0.4], [0.6,0.6,1.4])
                env.p_goal[:] = g_cur
                if env._goal_mid != -1:
                    env.data.mocap_pos[env._goal_mid] = env.p_goal

            # (2) 충돌 감지(손 제외)
            collided = False
            if hasattr(env,"_arm_gids") and hasattr(env,"_protect_gids"):
                arm_gids = env._arm_gids; prot_gids = env._protect_gids
                for i in range(env.data.ncon):
                    c = env.data.contact[i]
                    b1 = _mj.mj_id2name(env.model, _mj.mjtObj.mjOBJ_BODY, int(env.model.geom_bodyid[c.geom1]))
                    b2 = _mj.mj_id2name(env.model, _mj.mjtObj.mjOBJ_BODY, int(env.model.geom_bodyid[c.geom2]))
                    if (b1 and "hand" in b1) or (b2 and "hand" in b2): continue
                    if (c.geom1 in arm_gids and c.geom2 in prot_gids) or (c.geom2 in arm_gids and c.geom1 in prot_gids):
                        if max(0.0, -c.dist) >= 0.002: collided = True; break
            if collided: collision_freeze = True

            # (3) 현재 손과 g_cur 사이 거리
            p_hand = env._fk()
            dist = float(np.linalg.norm(env.p_goal - p_hand))

            # (4) 홀드/해제 논리
            if collision_freeze:
                hold_mode = True
            else:
                if not hold_mode and dist < max(0.8*reach_eps, 0.02):
                    hold_mode = True
                elif hold_mode and dist > (reach_eps + 0.02):
                    hold_mode = False; collision_freeze = False

            # (5) 스텝
            if hold_mode:
                action = np.zeros(env.action_space.shape, dtype=np.float32)
                obs, _, _, _, _ = env.step(action)
                env._step_count = 0
            else:
                action, _ = policy.predict(obs, deterministic=True)
                action = np.clip(action * speed.mult, env.action_space.low, env.action_space.high)
                obs, _, done, _, _ = env.step(action)
                if not collided: last_safe_qpos = env.data.qpos.copy()
                if done:
                    obs, _ = env.reset()
                    if last_safe_qpos is not None:
                        env.data.qpos[:] = last_safe_qpos; env.data.qvel[:] = 0.0
                        _mj.mj_forward(env.model, env.data)
                    env.p_goal[:] = g_cur
                    if env._goal_mid != -1:
                        env.data.mocap_pos[env._goal_mid] = env.p_goal

            # (6) 상태 출력(0.5초마다)
            now = time.time()
            if now - last_status > 0.5:
                print(f"[goto] dist={dist:.3f}m | goal_step={step if dist_goal>1e-6 else 0:.3f}m | hold={hold_mode} | collided={collision_freeze}")
                last_status = now

            # (7) 로봇 송신 ≤50Hz
            if robot.ok and (now - last_robot_send) > 0.02:
                qpos = {idx: float(env.data.qpos[adr]) for idx, adr in motor_qadr.items()}
                robot.send_qpos(qpos); last_robot_send = now

            # (8) 종료 조건: 최종 목표 근방 + 중간목표가 거의 도달
            if dist <= reach_eps and dist_goal < 1e-3:
                print(f"[goto] ✔ Reached goal: dist={dist:.3f} m (≤ {reach_eps:.3f})")
                break
            if (time.time() - start_t) > timeout:
                print(f"[goto] ✖ Timeout after {timeout:.1f}s (dist={dist:.3f} m)")
                break

            time.sleep(speed.dt)

    except KeyboardInterrupt:
        print("\n[goto] Aborted by user.")

    if robot.ok:
        qpos = {idx: float(env.data.qpos[adr]) for idx, adr in motor_qadr.items()}
        robot.send_qpos(qpos)
    print("[goto] Done.")

def PathLike(p: str):
    import pathlib
    return pathlib.Path(p).expanduser()

# --------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Detect blue ball (x,y,z) once and move G1 arm with slow approach.")
    # detector
    ap.add_argument("--pitch-deg", type=float, default=42.4)
    ap.add_argument("--cam-height", type=float, default=1.25)
    ap.add_argument("--flip-lr", action="store_true")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--show-hold-ms", type=int, default=800)
    ap.add_argument("--no-show-keep", action="store_true", help="검출 후 창 유지 끄기(기본: 유지)")
    ap.add_argument("--detect-timeout", type=float, default=15.0)
    # runner
    ap.add_argument("--model", default="models/ppo_g1_left_53178k.zip")
    ap.add_argument("--right-arm", action="store_true")
    ap.add_argument("--iface", default="eth0")
    ap.add_argument("--domain", type=int, default=0)
    ap.add_argument("--rate", type=float, default=0.04)
    ap.add_argument("--reach-eps", type=float, default=0.03)
    ap.add_argument("--timeout", type=float, default=20.0)
    ap.add_argument("--action-mult", type=float, default=1.0)
    ap.add_argument("--no-prompt", action="store_true")
    ap.add_argument("--approach-speed", type=float, default=0.05, help="intermediate goal speed [m/s]")

    args = ap.parse_args()

    dcfg = DetectorConfig(
        pitch_deg=args.pitch_deg,
        cam_height_m=args.cam_height,
        flip_lr=args.flip_lr,
        show=args.show,
        timeout_s=args.detect_timeout,
        show_hold_ms=args.show_hold_ms,
        show_keep=(not args.no_show_keep),
    )

    print("[main] Starting RealSense detector…")
    det = RealSenseBlueBallDetector(dcfg)
    det.start()
    try:
        x,y,z = det.get_xyz_once()
        print(f"[main] Detected goal: x={x:.3f} y={y:.3f} z={z:.3f} (m)")
    finally:
        det.stop()

    # 최종 안전 재클램프(이중 안전)
    x = min(max(0.0, x), X_MAX_M)
    z = max(Z_MIN_M, z)
    goal = np.array([x,y,z], dtype=np.float32)

    run_goto(goal=goal,
             model_path=args.model,
             iface=args.iface,
             domain=args.domain,
             right_arm=args.right_arm,
             rate=args.rate,
             reach_eps=args.reach_eps,
             timeout=args.timeout,
             action_mult=args.action_mult,
             no_prompt=args.no_prompt,
             approach_speed=max(0.0, float(args.approach_speed)))

if __name__ == "__main__":
    main()
