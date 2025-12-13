#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import math

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist


class Mission4GapFollow:
    """
    미션4: 라바콘 복도 주행용 하이브리드 (개선 버전)

    개선점
    1) 좌/우 거리 mean 대신 "유효 거리만" 추려 percentile(가까운 편)로 대표값 사용
       -> range_max(10m) 섞여 평균이 7m대로 붕괴하는 문제 해결
    2) wall_error(미터)를 steering 각도(rad)로 변환하여 goal_angle(rad)과 단위 일치
    3) 한쪽 벽만 보일 때는 wall 보정 비중을 낮춰서 흔들림/오동작 감소
    4) cmd_vel 토픽을 파라미터화(~cmd_topic)
    """

    def __init__(self):
        rospy.init_node("mission4_gap_follow")

        # ===== 0. 토픽 =====
        self.scan_topic = rospy.get_param("~scan_topic", "/scan")
        self.cmd_topic  = rospy.get_param("~cmd_topic",  "/cmd_vel")

        # ===== 1. 속도 관련 =====
        self.base_speed = rospy.get_param("~base_speed", 0.20)
        self.slow_speed = rospy.get_param("~slow_speed", 0.10)

        self.stop_dist = rospy.get_param("~stop_dist", 0.15)   # 이 이하면 정지
        self.slow_dist = rospy.get_param("~slow_dist", 0.70)   # 이 이하면 감속

        # ===== 2. 조향 관련 =====
        self.kp_steer   = rospy.get_param("~kp_steer", 1.2)

        # steering low-pass: 0이면 즉응, 0.6~0.75 추천
        self.lp_alpha   = rospy.get_param("~lp_alpha", 0.70)

        # LIMO 회전 방향 반전이 필요하면 -1.0, 아니면 1.0
        self.steer_sign = rospy.get_param("~steer_sign", 1.0)

        # gap vs 벽 정렬 비중 (기본: gap 조금 더)
        self.w_gap  = rospy.get_param("~w_gap", 0.60)
        self.w_wall = rospy.get_param("~w_wall", 0.40)

        # bias는 "각도(rad)"로 사용 (예: -0.03rad ≈ -1.7deg)
        self.bias_angle = rospy.get_param("~bias_angle", 0.0)

        # wall_angle 계산용 파라미터
        self.lookahead = rospy.get_param("~lookahead", 0.80)   # m
        self.k_wall    = rospy.get_param("~k_wall", 1.30)      # wall error gain

        # 한쪽 벽만 보일 때 wall 가중치 줄이기
        self.one_side_wall_scale = rospy.get_param("~one_side_wall_scale", 0.30)

        # ===== 3. 시야각 설정 =====
        # gap 탐색 범위
        self.gap_deg_min = rospy.get_param("~gap_deg_min", -70.0)
        self.gap_deg_max = rospy.get_param("~gap_deg_max",  70.0)

        # 전방 충돌 체크 범위
        self.front_deg_min = rospy.get_param("~front_deg_min", -35.0)
        self.front_deg_max = rospy.get_param("~front_deg_max",  35.0)

        # 좌/우 벽 정렬용 범위
        self.left_wall_deg_min  = rospy.get_param("~left_wall_deg_min",  20.0)
        self.left_wall_deg_max  = rospy.get_param("~left_wall_deg_max",  70.0)
        self.right_wall_deg_min = rospy.get_param("~right_wall_deg_min", -70.0)
        self.right_wall_deg_max = rospy.get_param("~right_wall_deg_max", -20.0)

        # ===== 4. FGM 파라미터 =====
        self.bubble_radius     = rospy.get_param("~bubble_radius", 0.30)
        self.expand_points     = rospy.get_param("~expand_points", 5)
        self.min_gap_width_deg = rospy.get_param("~min_gap_width_deg", 8.0)

        # ===== 5. 좌/우 거리 robust 파라미터 =====
        # "벽/콘 후보"로 볼 최대 거리(이보다 멀면 무시)
        self.wall_max = rospy.get_param("~wall_max", 2.5)
        # percentile: 10~30 추천 (낮을수록 더 '가까운 콘'에 민감)
        self.wall_pct = rospy.get_param("~wall_pct", 20)

        # ===== 6. 내부 상태 =====
        self.angle_min = None
        self.angle_max = None
        self.angle_inc = None
        self.prev_steer = 0.0

        # ===== 7. ROS 통신 =====
        self.sub_scan = rospy.Subscriber(self.scan_topic, LaserScan, self.scan_callback, queue_size=1)
        self.pub_cmd  = rospy.Publisher(self.cmd_topic, Twist, queue_size=1)

        rospy.on_shutdown(self.on_shutdown)
        rospy.loginfo("[M4-Hybrid-Center] node started. scan=%s cmd=%s", self.scan_topic, self.cmd_topic)

    # ---------- 유틸 ---------- #

    def _init_laser_meta(self, scan):
        self.angle_min = scan.angle_min
        self.angle_max = scan.angle_max
        self.angle_inc = scan.angle_increment
        rospy.loginfo("[M4] Laser meta: min=%.2f, max=%.2f, inc=%.4f",
                      self.angle_min, self.angle_max, self.angle_inc)

    def _deg_to_index_range(self, deg_min, deg_max):
        if self.angle_min is None or self.angle_inc is None:
            return None, None

        rad_min = math.radians(deg_min)
        rad_max = math.radians(deg_max)

        idx_min = int((rad_min - self.angle_min) / self.angle_inc)
        idx_max = int((rad_max - self.angle_min) / self.angle_inc)

        scan_len = int((self.angle_max - self.angle_min) / self.angle_inc)
        idx_min = max(0, min(idx_min, scan_len - 1))
        idx_max = max(0, min(idx_max, scan_len - 1))
        if idx_max < idx_min:
            idx_min, idx_max = idx_max, idx_min

        return idx_min, idx_max

    def _get_min_dist(self, ranges_np, deg_min, deg_max):
        idx_min, idx_max = self._deg_to_index_range(deg_min, deg_max)
        if idx_min is None:
            return None
        seg = ranges_np[idx_min:idx_max + 1]
        if len(seg) == 0:
            return None
        return float(np.min(seg))

    def _get_side_dist_robust(self, ranges_np, deg_min, deg_max, wall_max=None, pct=None):
        """range_max 같은 큰 값은 버리고, 가까운 값들 중 percentile로 대표 거리 추출"""
        if wall_max is None:
            wall_max = self.wall_max
        if pct is None:
            pct = self.wall_pct

        idx_min, idx_max = self._deg_to_index_range(deg_min, deg_max)
        if idx_min is None:
            return None, 0

        seg = ranges_np[idx_min:idx_max + 1]
        if len(seg) == 0:
            return None, 0

        valid = seg[np.isfinite(seg)]
        valid = valid[valid < wall_max]
        if len(valid) < 5:
            return None, len(valid)

        return float(np.percentile(valid, pct)), len(valid)

    # ---------- FGM: gap 방향 찾기 ---------- #

    def _find_best_direction(self, ranges_np):
        idx_min, idx_max = self._deg_to_index_range(self.gap_deg_min, self.gap_deg_max)
        if idx_min is None:
            return 0.0

        seg = ranges_np[idx_min:idx_max + 1].copy()
        n = len(seg)
        if n == 0:
            return 0.0

        danger = seg < self.bubble_radius

        # inflate
        if self.expand_points > 0:
            danger_indices = np.where(danger)[0]
            expanded = danger.copy()
            for i in danger_indices:
                s = max(0, i - self.expand_points)
                e = min(n - 1, i + self.expand_points)
                expanded[s:e + 1] = True
            danger = expanded

        # gaps
        gaps = []
        in_gap = False
        start_i = 0
        for i in range(n):
            if not danger[i]:
                if not in_gap:
                    in_gap = True
                    start_i = i
            else:
                if in_gap:
                    in_gap = False
                    gaps.append((start_i, i - 1))
        if in_gap:
            gaps.append((start_i, n - 1))

        if not gaps:
            return 0.0

        best_gap = None
        best_score = -1e9
        for (g_s, g_e) in gaps:
            width = g_e - g_s
            if width < 3:
                continue

            width_deg = abs(math.degrees(width * self.angle_inc))
            if width_deg < self.min_gap_width_deg:
                continue

            mid = (g_s + g_e) / 2.0
            center_offset = abs(mid - (n / 2.0))

            score = width - 0.12 * center_offset
            if score > best_score:
                best_score = score
                best_gap = (g_s, g_e)

        if best_gap is None:
            best_gap = max(gaps, key=lambda g: g[1] - g[0])

        g_s, g_e = best_gap
        center_idx_local = (g_s + g_e) / 2.0
        center_idx_global = idx_min + center_idx_local

        best_angle = self.angle_min + center_idx_global * self.angle_inc
        return best_angle  # rad

    # ---------- 콜백 ---------- #

    def scan_callback(self, scan):
        if self.angle_min is None:
            self._init_laser_meta(scan)

        ranges = np.array(scan.ranges, dtype=np.float32)

        max_r = scan.range_max if scan.range_max > 0 else 10.0
        min_r = scan.range_min if scan.range_min > 0 else 0.05

        # 전처리: 0 / NaN / inf -> max_r (측정 없음)
        ranges[~np.isfinite(ranges)] = max_r
        ranges[ranges <= 0.0] = max_r
        ranges = np.clip(ranges, min_r, max_r)

        # 1) FGM: gap 방향 (rad)
        goal_angle = self._find_best_direction(ranges)
        goal_deg = math.degrees(goal_angle)

        # 2) 좌/우 벽 robust 거리
        left_dist,  nL = self._get_side_dist_robust(ranges, self.left_wall_deg_min,  self.left_wall_deg_max)
        right_dist, nR = self._get_side_dist_robust(ranges, self.right_wall_deg_min, self.right_wall_deg_max)

        # 3) 중앙 정렬 오차(무차원) -> wall_angle(rad)
        wall_angle = 0.0
        wall_ok = (left_dist is not None) and (right_dist is not None)

        if wall_ok:
            err = (left_dist - right_dist) / (left_dist + right_dist + 1e-6)  # -1~1 근처
            wall_angle = math.atan2(self.k_wall * err, self.lookahead)
        else:
            # 한쪽만 보이면 벽 정렬 신뢰 낮음 -> wall_angle 약하게(혹은 0)
            wall_angle = 0.0

        # 4) 가중치 조정
        w_wall = self.w_wall
        if not wall_ok:
            w_wall *= self.one_side_wall_scale

        # 5) 최종 조향(rad) (단위 일치)
        steer_raw = self.w_gap * goal_angle + w_wall * wall_angle + self.bias_angle

        # low-pass
        steer = self.lp_alpha * self.prev_steer + (1.0 - self.lp_alpha) * steer_raw
        self.prev_steer = steer

        # 6) 전방 최소거리로 속도
        front_dist = self._get_min_dist(ranges, self.front_deg_min, self.front_deg_max)
        if front_dist is None:
            front_dist = max_r

        if front_dist <= self.stop_dist:
            linear_x = 0.0
        elif front_dist <= self.slow_dist:
            linear_x = self.slow_speed
        else:
            linear_x = self.base_speed

        # 7) 명령 발행
        cmd = Twist()
        cmd.linear.x = linear_x
        cmd.angular.z = self.steer_sign * self.kp_steer * steer
        self.pub_cmd.publish(cmd)

        rospy.loginfo_throttle(
            0.5,
            "[M4] front=%.2f goal=%.1fdeg wall=%.1fdeg L=%.2f(%d) R=%.2f(%d) vx=%.2f wz=%.2f",
            front_dist,
            goal_deg,
            math.degrees(wall_angle),
            left_dist if left_dist is not None else -1.0, nL,
            right_dist if right_dist is not None else -1.0, nR,
            linear_x,
            cmd.angular.z
        )

    def on_shutdown(self):
        self.pub_cmd.publish(Twist())
        rospy.loginfo("[M4] Shutting down (stop).")


if __name__ == "__main__":
    try:
        Mission4GapFollow()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

