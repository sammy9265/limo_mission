#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
import math
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage

class LimoMissionUnified:
    def __init__(self):
        rospy.init_node("limo_mission_unified", anonymous=False)

        # =========================
        # Pub/Sub
        # =========================
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)

        self.use_compressed = rospy.get_param("~use_compressed", True)
        if self.use_compressed:
            self.img_sub = rospy.Subscriber(
                "/usb_cam/image_raw/compressed", CompressedImage, self.image_callback, queue_size=1
            )
        else:
            self.img_sub = rospy.Subscriber(
                "/usb_cam/image_raw", Image, self.image_callback, queue_size=1
            )

        self.scan_sub = rospy.Subscriber("/scan", LaserScan, self.scan_callback, queue_size=1)

        # =========================
        # State
        # =========================
        # 순서: LINE -> GAP(장애물) -> M4(라바콘) -> EDGE_LANE(검은트랙)
        self.mode = "LINE"
        self.last_mode_change = rospy.Time.now().to_sec()

        # =========================
        # (A) LINE (미션1/2) 파라미터
        # =========================
        self.roi_row = 0.6
        self.black_threshold = 75
        self.line_speed = 0.18
        self.line_kp = 0.005
        self.line_deadzone = 15

        self.line_seen = False
        self.line_seen_cnt = 0
        self.line_seen_enter = 6
        self.line_lost_cnt = 0

        # =========================
        # (B) 라이다 장애물 트리거(미션3 진입)
        # =========================
        self.trigger_dist = 1.0
        self.trigger_count = 0
        self.trigger_limit = 5

        # GAP->LINE 복귀 조건 (M4 진입 전까지만 유효)
        self.front_clear_dist = 1.2
        self.front_clear_cnt = 0
        self.front_clear_enter = 8

        # =========================
        # (C) GAP follower (미션3)
        # =========================
        self.gap_speed = 0.22
        self.gap_kp = 1.5
        self.safe_dist = 1.0
        self.view_limit_left = 45.0 
        self.view_limit_right = -80.0

        # =========================
        # (D) 코너 충돌/후진(RECOVER)
        # =========================
        self.corner_safe_dist_default = 0.12
        self.corner_safe_dist_narrow = 0.12
        self.wall_ignore_y = 0.5
        self.recover_start_time = 0.0
        self.recover_duration = 2.0
        self.recover_dir = 1
        self.recover_rot_speed = 0.7
        self.corner_enter_cnt = 3
        self.left_corner_cnt = 0
        self.right_corner_cnt = 0
        self.recover_cooldown = 1.0
        self.recover_cooldown_until = 0.0
        self.corner_need_front = 0.17
        self.front_hard_stop = 0.13

        # =========================
        # (E) M4 Priority Avoid (카메라 기반 - 라바콘)
        # =========================
        self.img_width = 320
        self.img_height = 240
        self.m4_speed = 0.20
        self.roi_ratio = 0.4
        self.roi_h = int(self.img_height * self.roi_ratio)

        self.lower_white = np.array([0, 0, 160])
        self.upper_white = np.array([179, 30, 255])
        self.lower_red1 = np.array([0, 100, 50])
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([170, 100, 50])
        self.upper_red2 = np.array([180, 255, 255])

        self.gain_white = 0.005
        self.gain_red = 0.008
        self.detect_thresh = 50

        self.avoid_level = 0
        self.avoid_cnt = 0
        self.avoid_enter = 6
        self.avoid_exit = 10
        
        # M4 완료(라바콘 끝남) 감지용
        self.m4_clear_count = 0 
        self.m4_clear_limit = 20  # 약 1~2초간 빨간색 안보이면 다음 미션으로

        # =========================
        # [NEW] (F) Edge Lane (미션 5/6 - 검은 트랙 추종)
        # 요청하신 두 번째 코드를 위한 파라미터
        # =========================
        self.edge_forward_speed = 0.12
        self.edge_search_spin_speed = 0.25
        self.edge_k_angle = 0.010
        self.edge_dark_ratio = 0.3
        self.edge_min_pixels = 5

        # =========================
        # 공유 결과
        # =========================
        self.line_twist = Twist()
        self.gap_twist = Twist()
        self.m4_twist = Twist()
        self.edge_twist = Twist() # 새로 추가됨

        self.have_scan = False
        self.front_min = 10.0
        self.path_clear_dist = 10.0
        self.left_corner_min = 10.0
        self.right_corner_min = 10.0

        self.control_timer = rospy.Timer(rospy.Duration(0.05), self.control_loop)

        rospy.loginfo("Unified node started (LINE->GAP->M4->EDGE).")

    # -------------------------
    # Camera decode helper
    # -------------------------
    def decode_image(self, msg):
        if self.use_compressed:
            np_arr = np.frombuffer(msg.data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            return img
        else:
            try:
                buf = np.frombuffer(msg.data, dtype=np.uint8)
                img = buf.reshape(msg.height, msg.step)[:, :msg.width * 3]
                img = img.reshape(msg.height, msg.width, 3)
                return img
            except Exception:
                return None

    # ---------------------------------------------------
    # [NEW] 요청하신 두 번째 코드 로직을 함수로 변환
    # EdgeLaneNoBridge의 image_callback 로직
    # ---------------------------------------------------
    def process_edge_lane(self, img):
        t = Twist()
        # 원본 해상도 사용 (리사이즈 된거 말고 원본 msg에서 온거 쓰거나, 여기서 다시 처리)
        # 여기서는 편의상 self.img_width(320x240) 리사이즈 된 것을 쓰면 파라미터 튜닝이 달라질 수 있음.
        # 기존 코드가 원본 해상도(640x480 등) 기준일 수 있으나, 
        # 로직이 비율 기반(offset_norm)이라 320x240에서도 작동할 것임.

        h, w, _ = img.shape
        center = w / 2.0

        # 1) 바닥 쪽 ROI (하단 50%)
        roi_y_start = int(h * 0.5)
        roi = img[roi_y_start:, :]

        # 2) 그레이 + 블러
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # 3) 검은 트랙 강조: THRESH_BINARY_INV + OTSU
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 4) 노이즈 제거
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # 5) 열별 픽셀 수 계산
        mask = (binary > 0)
        col_sum = np.sum(mask, axis=0)
        max_val = int(np.max(col_sum)) if col_sum.size > 0 else 0

        # 트랙 못 찾음 -> 회전
        if max_val < self.edge_min_pixels:
            t.linear.x = 0.0
            t.angular.z = self.edge_search_spin_speed
            return t

        # 6) 후보 열 추출
        threshold_val = max(self.edge_min_pixels, int(max_val * self.edge_dark_ratio))
        candidates = np.where(col_sum >= threshold_val)[0]

        if candidates.size == 0:
            t.linear.x = 0.0
            t.angular.z = self.edge_search_spin_speed
            return t

        # 7) 무게중심 계산
        x = np.arange(len(col_sum))
        numerator = np.sum(x[candidates] * col_sum[candidates])
        denominator = np.sum(col_sum[candidates])
        
        if denominator == 0:
            t.linear.x = 0.0
            t.angular.z = self.edge_search_spin_speed
            return t

        track_center_x = float(numerator / denominator)
        offset = track_center_x - center 
        
        # 조향 계산
        ang = -self.edge_k_angle * offset
        ang = max(min(ang, 0.8), -0.8)

        t.linear.x = self.edge_forward_speed
        t.angular.z = ang
        
        return t

    # -------------------------
    # Camera callback
    # -------------------------
    def image_callback(self, msg):
        img_raw = self.decode_image(msg)
        if img_raw is None:
            return

        # 리사이즈 (기존 로직용)
        img = cv2.resize(img_raw, (self.img_width, self.img_height))

        # ===== (1) LINE controller (미션 1,2용) =====
        h, w, _ = img.shape
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        roi = gray[int(h * self.roi_row):h, :]
        _, mask = cv2.threshold(roi, self.black_threshold, 255, cv2.THRESH_BINARY_INV)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

        M = cv2.moments(mask)
        t = Twist()
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            error = cx - (w / 2.0)
            if abs(error) < self.line_deadzone:
                error = 0.0
            t.linear.x = self.line_speed
            t.angular.z = -(self.line_kp * error)
            self.line_seen = True
        else:
            t.linear.x = self.line_speed
            t.angular.z = 0.0
            self.line_seen = False
        self.line_twist = t

        if self.line_seen:
            self.line_seen_cnt += 1
            self.line_lost_cnt = 0
        else:
            self.line_lost_cnt += 1
            self.line_seen_cnt = 0

        # ===== (2) M4 priority avoid controller (미션 4용) =====
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        roi_hsv = hsv[self.img_height - self.roi_h:, :]

        mask_r1 = cv2.inRange(roi_hsv, self.lower_red1, self.upper_red1)
        mask_r2 = cv2.inRange(roi_hsv, self.lower_red2, self.upper_red2)
        mask_red = cv2.bitwise_or(mask_r1, mask_r2)
        mask_white = cv2.inRange(roi_hsv, self.lower_white, self.upper_white)

        cnt_red = cv2.countNonZero(mask_red)
        cnt_white = cv2.countNonZero(mask_white)

        h2, w2 = roi_hsv.shape[:2]
        cx2 = w2 // 2

        m4 = Twist()
        m4.linear.x = self.m4_speed

        level = 0
        # 빨간색이 보이면 회피 로직
        if cnt_red > self.detect_thresh:
            left_mass = cv2.countNonZero(mask_red[:, :cx2])
            right_mass = cv2.countNonZero(mask_red[:, cx2:])
            error = right_mass - left_mass
            m4.angular.z = error * self.gain_red
            level = 2
            self.m4_clear_count = 0 # 빨간색이 보이면 초기화
        elif cnt_white > self.detect_thresh:
            left_mass = cv2.countNonZero(mask_white[:, :cx2])
            right_mass = cv2.countNonZero(mask_white[:, cx2:])
            error = right_mass - left_mass
            m4.angular.z = error * self.gain_white
            level = 1
        else:
            m4.angular.z = 0.0
            level = 0
            # M4 모드일 때 빨간색이 안 보이는 횟수 증가
            if self.mode == "M4":
                self.m4_clear_count += 1

        m4.angular.z = float(np.clip(m4.angular.z, -1.5, 1.5))
        self.m4_twist = m4
        self.avoid_level = level

        if self.avoid_level > 0:
            self.avoid_cnt += 1
        else:
            self.avoid_cnt -= 1
        self.avoid_cnt = int(np.clip(self.avoid_cnt, 0, 50))

        # ===== (3) [NEW] Edge Lane Controller (미션 5/6용) =====
        # 여기서는 img_raw(원본) 또는 img(리사이즈)를 넘겨서 처리
        self.edge_twist = self.process_edge_lane(img)

    # -------------------------
    # LiDAR callback
    # -------------------------
    def scan_callback(self, msg):
        self.have_scan = True
        ranges = msg.ranges
        angle_min = msg.angle_min
        angle_inc = msg.angle_increment

        front_min = 10.0
        path_clear_dist = 10.0
        left_corner_min = 10.0
        right_corner_min = 10.0
        scan_data = []

        for i, r in enumerate(ranges):
            if np.isinf(r) or r < 0.1:
                continue

            raw_angle = angle_min + i * angle_inc
            robot_angle = raw_angle
            deg = math.degrees(robot_angle)

            if deg < -60:
                if r > 0.9: r = 0.9

            if abs(deg) < 10:
                if r < front_min: front_min = r
                if r < path_clear_dist: path_clear_dist = r

            if abs(deg) > 90: continue

            if self.view_limit_right < deg < self.view_limit_left:
                scan_data.append((robot_angle, min(r, 3.0)))

            oy = r * math.sin(robot_angle)
            if abs(oy) > self.wall_ignore_y: continue

            if 20 < deg < 60:
                if r < left_corner_min: left_corner_min = r
            elif -60 < deg < -20:
                if r < right_corner_min: right_corner_min = r

        self.front_min = front_min
        self.path_clear_dist = path_clear_dist
        self.left_corner_min = left_corner_min
        self.right_corner_min = right_corner_min

        # (LINE 모드일 때) 장애물 트리거 -> GAP 모드(미션3) 진입
        if self.mode == "LINE":
            if self.front_min < self.trigger_dist:
                self.trigger_count += 1
            else:
                self.trigger_count = 0

        # GAP 컨트롤 계산
        scan_data.sort(key=lambda x: x[0])
        best_gap_center = 0.0
        max_gap_len = 0
        current_start = -1
        current_len = 0

        for i in range(len(scan_data)):
            _, dist = scan_data[i]
            if dist > self.safe_dist:
                if current_start == -1: current_start = i
                current_len += 1
            else:
                if current_start != -1:
                    if current_len > max_gap_len:
                        max_gap_len = current_len
                        s, e = current_start, i - 1
                        best_gap_center = scan_data[int((s + e) / 2)][0]
                    current_start = -1
                    current_len = 0
        if current_start != -1 and current_len > max_gap_len:
            s, e = current_start, len(scan_data) - 1
            best_gap_center = scan_data[int((s + e) / 2)][0]
            max_gap_len = current_len

        t = Twist()
        if max_gap_len > 5:
            t.linear.x = self.gap_speed
            t.angular.z = self.gap_kp * best_gap_center
            if self.right_corner_min < 0.30: t.angular.z += 0.3
            elif self.left_corner_min < 0.30: t.angular.z -= 0.3
            t.angular.z = float(np.clip(t.angular.z, -1.2, 1.2))
        else:
            t.linear.x = 0.0
            t.angular.z = -0.6
        self.gap_twist = t

    # -------------------------
    # control loop
    # -------------------------
    def control_loop(self, _evt):
        if not self.have_scan:
            self.cmd_pub.publish(Twist())
            return

        now = rospy.Time.now().to_sec()

        # [RECOVER 모드]
        if self.mode == "RECOVER":
            elapsed = now - self.recover_start_time
            if elapsed > self.recover_duration:
                # RECOVER 끝나면 GAP으로 복귀 (M4 중이었어도 안전을 위해 GAP으로)
                self.mode = "GAP"
                self.last_mode_change = now
                self.recover_cooldown_until = now + self.recover_cooldown
                return
            t = Twist()
            if elapsed < 0.5:
                t.linear.x = -0.20
                t.angular.z = 0.0
            else:
                t.linear.x = 0.0
                rot = self.recover_rot_speed
                t.angular.z = rot if self.recover_dir == -1 else -rot
            self.cmd_pub.publish(t)
            return

        # [코너 충돌 감지 -> RECOVER 진입] (EDGE_LANE 모드 아닐때만)
        # EDGE_LANE 모드는 트랙만 보고 가야 하므로 여기서 간섭 안하게 할지, 안전장치로 둘지 결정.
        # 여기서는 EDGE_LANE 일때는 코너 감지 일단 끔 (트랙만 믿음)
        if self.mode != "EDGE_LANE":
            if now >= self.recover_cooldown_until:
                current_safe_limit = self.corner_safe_dist_default
                if self.path_clear_dist > 0.4:
                    current_safe_limit = self.corner_safe_dist_narrow

                danger_front = (self.front_min < self.front_hard_stop)
                allow_corner = (self.path_clear_dist < self.corner_need_front) or danger_front
                danger_left = allow_corner and (self.left_corner_min < current_safe_limit)
                danger_right = allow_corner and (self.right_corner_min < current_safe_limit)

                self.left_corner_cnt = self.left_corner_cnt + 1 if danger_left else 0
                self.right_corner_cnt = self.right_corner_cnt + 1 if danger_right else 0

                if self.left_corner_cnt >= self.corner_enter_cnt:
                    self.mode = "RECOVER"
                    self.recover_dir = 1
                    self.recover_start_time = now
                    self.last_mode_change = now
                    self.left_corner_cnt = 0
                    self.right_corner_cnt = 0
                    return
                if self.right_corner_cnt >= self.corner_enter_cnt:
                    self.mode = "RECOVER"
                    self.recover_dir = -1
                    self.recover_start_time = now
                    self.last_mode_change = now
                    self.left_corner_cnt = 0
                    self.right_corner_cnt = 0
                    return

        # ===== 상태 전환 로직 =====
        
        # 1. LINE -> GAP (장애물 감지시 미션3 진입)
        if self.mode == "LINE":
            if self.trigger_count >= self.trigger_limit:
                self.mode = "GAP"
                self.last_mode_change = now
                self.trigger_count = 0
                rospy.loginfo("SWITCH: LINE -> GAP")
                # 잠시 정지
                stop = Twist()
                for _ in range(5):
                    self.cmd_pub.publish(stop)
                    rospy.sleep(0.05)
                return

        # 2. GAP -> M4 (장애물 회피하다가 빨간색/흰색 라바콘 보이면 미션4 진입)
        if self.mode == "GAP":
            if self.avoid_cnt >= self.avoid_enter:
                self.mode = "M4"
                self.last_mode_change = now
                self.m4_clear_count = 0 # 초기화
                rospy.loginfo("SWITCH: GAP -> M4 (Red cone detected)")

        # 3. M4 -> EDGE_LANE (라바콘이 끝났다고 판단되면 미션5 진입)
        elif self.mode == "M4":
            # 빨간색이 일정 시간(m4_clear_limit) 이상 안 보이면 끝났다고 판단
            if self.m4_clear_count >= self.m4_clear_limit:
                self.mode = "EDGE_LANE"
                self.last_mode_change = now
                rospy.loginfo("SWITCH: M4 -> EDGE_LANE (Finished red cones)")
        
        # 참고: EDGE_LANE 모드는 마지막 미션이므로 다른 모드로 돌아가지 않음.

        # ===== 최종 Publish =====
        cmd = Twist()
        if self.mode == "LINE":
            cmd = self.line_twist
        elif self.mode == "GAP":
            cmd = self.gap_twist
            if self.front_min < 0.35:
                cmd.linear.x = 0.0
            elif self.front_min < 0.55:
                cmd.linear.x = min(cmd.linear.x, 0.08)
        elif self.mode == "M4":
            cmd = self.m4_twist
            if self.front_min < 0.6:
                cmd.linear.x = min(cmd.linear.x, 0.12)
        elif self.mode == "EDGE_LANE":
            cmd = self.edge_twist
            # 여기에는 별도의 라이다 정지 로직을 넣지 않았으나 필요하면 추가 가능
        else:
            cmd = Twist()

        self.cmd_pub.publish(cmd)

if __name__ == "__main__":
    try:
        LimoMissionUnified()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
