import math
from collections import deque

import cv2
import numpy as np
import pyrealsense2 as rs

# ===================== 설정 =====================

# 카메라 '아래로' 기울어진 각도(deg)
PITCH_DEG = 42.4

# 카메라 광학 중심의 바닥 기준 높이(m)
CAMERA_HEIGHT_M = 1.25

# 좌우 부호 (기본: 오른쪽 +). 왼쪽을 +로 쓰려면 True
FLIP_LR = False

# 파란색 HSV 범위(환경에 맞게 조정 필요)
LOWER_BLUE = np.array([95, 120, 60])
UPPER_BLUE = np.array([130, 255, 255])

# 전처리/노이즈
BLUR_KSIZE = 5
KERNEL = np.ones((5, 5), np.uint8)
MIN_AREA = 300              # 너무 작은 잡음 제거

# 깊이/좌표 안정화
DEPTH_WIN = 5               # 중심 주변 depth 중앙값 창
SMOOTH_WIN = 5              # 좌표 이동평균

# 멀티포인트 deprojection
USE_MULTIPOINT = True
MULTI_RADIUS = 7
SAMPLE_MAX = 200

# 공 지름이 5 cm → 반지름 0.025 m (표면→중심 보정)
BALL_RADIUS_M = 0.025

# =================================================


def median_depth(depth_m: np.ndarray, u: float, v: float, win: int = DEPTH_WIN) -> float:
    H, W = depth_m.shape
    u0 = max(0, int(u) - win // 2)
    v0 = max(0, int(v) - win // 2)
    u1 = min(W, u0 + win)
    v1 = min(H, v0 + win)
    patch = depth_m[v0:v1, u0:u1].astype(np.float32)
    patch = patch[patch > 0]
    if patch.size == 0:
        return 0.0
    return float(np.median(patch))


def apply_depth_filters(depth_frame: rs.depth_frame) -> rs.depth_frame:
    spatial = rs.spatial_filter()
    temporal = rs.temporal_filter()
    hole_filling = rs.hole_filling_filter()
    f = depth_frame
    f = spatial.process(f)
    f = temporal.process(f)
    f = hole_filling.process(f)
    return f


def deproject_multipoint(mask: np.ndarray, depth_m: np.ndarray,
                         intr: rs.intrinsics, cx: float, cy: float,
                         radius: int, sample_max: int):
    Hm, Wm = mask.shape
    Hd, Wd = depth_m.shape

    y0 = max(0, int(cy) - radius)
    y1 = min(Hm, int(cy) + radius + 1)
    x0 = max(0, int(cx) - radius)
    x1 = min(Wm, int(cx) + radius + 1)

    sub_mask = mask[y0:y1, x0:x1] > 0
    ys, xs = np.where(sub_mask)
    if ys.size == 0:
        return [], [], []

    xs_full = (xs + x0).astype(np.int32)
    ys_full = (ys + y0).astype(np.int32)

    # 샘플 수 제한
    if xs_full.size > sample_max:
        step = max(1, xs_full.size // sample_max)
        xs_full = xs_full[::step]
        ys_full = ys_full[::step]

    ptsX, ptsY, ptsZ = [], [], []
    for u, v in zip(xs_full, ys_full):
        # 경계 체크(깊이 해상도 기준)
        if v < 0 or v >= Hd or u < 0 or u >= Wd:
            continue
        d = float(depth_m[v, u])
        if d <= 0.0:
            continue
        Xc, Yc, Zc = rs.rs2_deproject_pixel_to_point(intr, [float(u), float(v)], d)
        ptsX.append(Xc); ptsY.append(Yc); ptsZ.append(Zc)

    return ptsX, ptsY, ptsZ


def main():
    # ---------- RealSense 파이프라인 ----------
    pipeline = rs.pipeline()
    config = rs.config()
    # 640x480@30 권장 (color/depth 동일)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    align = rs.align(rs.stream.color)
    prof = pipeline.start(config)

    depth_sensor = prof.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()  # 보통 0.001(m)

    color_stream = prof.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_stream.get_intrinsics()  # fx, fy, ppx, ppy 포함

    theta = math.radians(PITCH_DEG)
    hist = deque(maxlen=SMOOTH_WIN)

    print("실행 중: ESC 또는 q 로 종료")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)

            depth_raw = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not depth_raw or not color_frame:
                continue

            depth_f = apply_depth_filters(depth_raw)
            depth_m = np.asanyarray(depth_f.get_data()).astype(np.float32) * depth_scale
            color = np.asanyarray(color_frame.get_data())

            # (디버그) 해상도 불일치 경고
            if depth_m.shape != color.shape[:2]:
                print(f"[WARN] depth {depth_m.shape} vs color {color.shape[:2]} 해상도 불일치")

            # --- 파란색 마스크 ---
            blurred = cv2.GaussianBlur(color, (BLUR_KSIZE, BLUR_KSIZE), 0)
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL, iterations=2)

            # --- 컨투어 / bbox / 중심 ---
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            out_text = "공 미검출"
            if cnts:
                cnt = max(cnts, key=cv2.contourArea)
                area = cv2.contourArea(cnt)
                if area >= MIN_AREA:
                    bx, by, bw, bh = cv2.boundingRect(cnt)

                    # 중심(모멘트 기반; 실패 시 bbox 중점)
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cx = M["m10"] / M["m00"]
                        cy = M["m01"] / M["m00"]
                    else:
                        cx = bx + bw / 2.0
                        cy = by + bh / 2.0

                    # ---- 깊이 & 3D ----
                    ptsX, ptsY, ptsZ = [], [], []
                    if USE_MULTIPOINT:
                        ptsX, ptsY, ptsZ = deproject_multipoint(
                            mask, depth_m, intr, cx, cy, MULTI_RADIUS, SAMPLE_MAX
                        )
                    if (not USE_MULTIPOINT) or len(ptsZ) == 0:
                        dep = median_depth(depth_m, cx, cy, DEPTH_WIN)
                        if dep > 0.0:
                            Xc, Yc, Zc = rs.rs2_deproject_pixel_to_point(
                                intr, [float(cx), float(cy)], float(dep)
                            )
                            ptsX = [Xc]; ptsY = [Yc]; ptsZ = [Zc]

                    if len(ptsZ) > 0:
                        # 카메라 좌표 중앙값
                        Xc = float(np.median(ptsX))   # right +
                        Yc = float(np.median(ptsY))   # down +
                        Zc = float(np.median(ptsZ))   # forward +

                        # ---- 피치 보정 (아래로 PITCH_DEG) ----
                        # 카메라 프레임 → (right, up, forward)
                        xr = Xc
                        yr = -Yc  # up +
                        zr = Zc

                        # x-축 회전 R_x(θ): 카메라 기울기 보정
                        y_corr = yr * math.cos(theta) - zr * math.sin(theta)
                        # z_corr는 현재 사용하지 않지만 필요시 계산 가능:
                        # z_corr = yr * math.sin(theta) + zr * math.cos(theta)
                        x_corr = xr

                        # ---- 좌표계 매핑 ----
                        # x: 보정 없이 원본 깊이에 공 반지름만 더함 (표면→중심 보정)
                        x = float(Zc) + BALL_RADIUS_M
                        # y: 좌우 (보정 적용)
                        y = float(-x_corr) if FLIP_LR else float(x_corr)
                        # z: 바닥 기준 높이 (보정된 위(+) + 카메라 높이)
                        z = CAMERA_HEIGHT_M + float(y_corr)

                        # 이동평균 (x,y,z 모두)
                        hist.append((x, y, z))
                        sx = float(np.mean([p[0] for p in hist]))
                        sy = float(np.mean([p[1] for p in hist]))
                        sz = float(np.mean([p[2] for p in hist]))

                        out_text = f"x={sx:.3f}m  y={sy:.3f}m  z={sz:.3f}m"

                        # ---- 시각화 ----
                        cv2.circle(color, (int(cx), int(cy)), 6, (0, 255, 255), -1)
                        cv2.rectangle(color, (int(bx), int(by)),
                                      (int(bx + bw), int(by + bh)), (255, 0, 0), 2)
                    else:
                        out_text = "깊이/포인트 부족: 각도·조명 조정"

            cv2.putText(color, out_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(color, out_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(color, f"Pitch: {PITCH_DEG:.1f}° | CamH: {CAMERA_HEIGHT_M:.2f}m | FLIP_LR={FLIP_LR}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 2, cv2.LINE_AA)
            cv2.putText(color, "Press 'q' or 'ESC' to quit",
                        (10, color.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240,240,240), 2, cv2.LINE_AA)

            cv2.imshow("Color", color)
            cv2.imshow("Mask", mask)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
