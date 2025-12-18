#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage, LaserScan
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

class LineTracerWithObstacleAvoidance:
    def __init__(self):
        rospy.init_node("line_tracer_with_obstacle_avoidance")
        self.pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)

        # ==========================================================
        # [수정됨] 카메라 토픽을 로봇 환경에 맞게 변경
        # /usb_cam/... -> /camera/rgb/...
        # ==========================================================
        rospy.Subscriber("/camera/rgb/image_raw/compressed", CompressedImage, self.camera_cb)
        rospy.Subscriber("/scan", LaserScan, self.lidar_cb)

        self.bridge = CvBridge()

        # ==========================================
        # [설정] 빨간색 회피 주행 파라미터 (NEW)
        # ==========================================
        self.red_thresh = 500          # 이 개수 이상 빨간점이 보이면 빨간색 모드 진입
        self.red_gain = 0.005          # 빨간색 회피 조향 게인
        self.red_speed = 0.12          # 빨간색 구간 속도

        # ==========================================
        # [설정] 검은색 라인트레이싱 파라미터 (기존 로직 변경)
        # ==========================================
        self.forward_speed = 0.12      # 기본 전진 속도
        self.search_spin_speed = 0.25  # 라인 못 찾을 때 회전 속도
        self.k_angle = 0.010           # 조향 게인
        self.dark_min_pixels = 5       # 최소 검은색 픽셀 수
        self.dark_col_ratio = 0.3      # 임계값 비율

        # ==========================================
        # [설정] LiDAR 및 상태 변수 (수정 금지)
        # ==========================================
        self.scan_ranges = []
        self.front = 999.0

        self.state = "LANE"
        self.escape_angle = 0.0
        self.state_start = rospy.Time.now().to_sec()

        self.left_escape_count = 0
        self.force_right_escape = 0
        self.robot_width = 0.13    

    # ============================================================
    # LIDAR (수정 금지)
    # ============================================================
    def lidar_cb(self, scan):
        raw = np.array(scan.ranges)
        self.scan_ranges = raw

        front_zone = np.concatenate([raw[:10], raw[-10:]])
        cleaned = [d for d in front_zone if d > 0.20 and not np.isnan(d)]
        self.front = np.median(cleaned) if cleaned else 999.0

    # ============================================================
    # CAMERA (우선순위: 빨간색 회피 -> 검은색 추종)
    # ============================================================
    def camera_cb(self, msg):
        twist = Twist()
        now = rospy.Time.now().to_sec()

        # 1. ESCAPE 모드 (기존 유지)
        if self.state == "ESCAPE":
            self.escape_control()
            return

        # 2. BACK 모드 (기존 유지)
        if self.state == "BACK":
            self.back_control()
            return

        # 3. LANE 모드
        if self.state == "LANE":

            # [LiDAR Check] 장애물 감지 시 BACK 모드 (최우선 - 수정 금지)
            if self.front < 0.45:
                self.state = "BACK"
                self.state_start = now
                return

            try:
                # 이미지 변환
                img = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
                h, w = img.shape[:2]
                center = w / 2.0

                # 전처리: Blur & HSV
                img_blur = cv2.GaussianBlur(img, (5, 5), 0)
                hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)

                # ROI 설정 (하단부 50%)
                roi_h = int(h * 0.5)
                roi_hsv = hsv[h - roi_h:, :]                     # 빨간색용 (HSV)
                roi_gray = cv2.cvtColor(img[h - roi_h:, :], cv2.COLOR_BGR2GRAY) # 검은색용 (Gray)

                # --------------------------------------------------
                # [PRIORITY 1] 빨간색 감지 및 회피 (Avoidance)
                # --------------------------------------------------
                lower_red1 = np.array([0, 100, 50]);  upper_red1 = np.array([10, 255, 255])
                lower_red2 = np.array([170, 100, 50]); upper_red2 = np.array([180, 255, 255])
                
                mask_r1 = cv2.inRange(roi_hsv, lower_red1, upper_red1)
                mask_r2 = cv2.inRange(roi_hsv, lower_red2, upper_red2)
                mask_red = cv2.bitwise_or(mask_r1, mask_r2)

                red_count = cv2.countNonZero(mask_red)

                # 빨간색이 일정량 이상 보이면 -> 빨간색 회피 로직 실행
                if red_count > self.red_thresh:
                    # 화면 좌우 분할하여 빨간색 양 비교
                    cx = w // 2
                    left_mass = cv2.countNonZero(mask_red[:, :cx])
                    right_mass = cv2.countNonZero(mask_red[:, cx:])

                    # [회피 논리]
                    # 오른쪽 빨강이 많음(양수) -> 왼쪽으로 회전(Turn Left, +z)
                    # 왼쪽 빨강이 많음(음수) -> 오른쪽으로 회전(Turn Right, -z)
                    error = right_mass - left_mass
                    
                    steer = error * self.red_gain
                    steer = float(np.clip(steer, -1.0, 1.0))

                    twist.linear.x = self.red_speed
                    twist.angular.z = steer
                    self.pub.publish(twist)
                    return # 빨간색 처리했으므로 검은색 로직 실행 안함 (return)

                # --------------------------------------------------
                # [PRIORITY 2] 검은색 트랙 추종 (Line Following)
                # --------------------------------------------------
                
                # 검은색 트랙 강조: THRESH_BINARY_INV + OTSU
                # (검은색 라인이 흰색(255)으로 변환됨)
                _, binary = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

                # 노이즈 제거
                kernel = np.ones((3, 3), np.uint8)
                binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
                binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

                # 열별 픽셀 수 계산 (Column Sum)
                mask_black = (binary > 0)
                col_sum = np.sum(mask_black, axis=0)
                max_val = int(np.max(col_sum)) if col_sum.size > 0 else 0

                # 검은색 라인이 너무 적으면 -> 못 찾음 -> 제자리 회전 (라인 탐색)
                if max_val < self.dark_min_pixels:
                    twist.linear.x = 0.0
                    twist.angular.z = self.search_spin_speed
                    self.pub.publish(twist)
                    return

                # 유효한 열 후보 추출
                threshold_val = max(self.dark_min_pixels, int(max_val * self.dark_col_ratio))
                candidates = np.where(col_sum >= threshold_val)[0]

                if candidates.size == 0:
                    twist.linear.x = 0.0
                    twist.angular.z = self.search_spin_speed
                    self.pub.publish(twist)
                    return

                # 무게중심(Center of Mass) 계산 - 검은 라인의 중심 찾기
                numerator = np.sum(candidates * col_sum[candidates])
                denominator = np.sum(col_sum[candidates])
                track_center_x = float(numerator / denominator)

                # [추종 논리]
                # 라인이 화면 중심보다 오른쪽(Positive offset) -> 오른쪽으로 회전해야 함 (-z)
                offset = track_center_x - center 
                ang = -self.k_angle * offset
                ang = max(min(ang, 0.8), -0.8)

                # 최종 주행 명령
                twist.linear.x = self.forward_speed
                twist.angular.z = ang
                self.pub.publish(twist)

            except Exception as e:
                rospy.logerr(f"Image Processing Error: {e}")

    # ============================================================
    # BACK MODE (수정 금지)
    # ============================================================
    def back_control(self):
        twist = Twist()
        now = rospy.Time.now().to_sec()

        if now - self.state_start < 1.4:
            twist.linear.x = -0.24
            twist.angular.z = 0.0
            self.pub.publish(twist)
        else:
            angle = self.find_gap_max()
            angle = self.apply_escape_direction_logic(angle)

            self.escape_angle = angle
            self.state = "ESCAPE"
            self.state_start = now

    # ============================================================
    # ESCAPE MODE (수정 금지)
    # ============================================================
    def escape_control(self):
        twist = Twist()
        now = rospy.Time.now().to_sec()

        if now - self.state_start < 1.0:
            twist.linear.x = 0.19
            twist.angular.z = self.escape_angle * 1.3
            self.pub.publish(twist)
        else:
            self.state = "LANE"

    # ============================================================
    # ESCAPE 방향 로직 (수정 금지)
    # ============================================================
    def apply_escape_direction_logic(self, angle):
        if self.force_right_escape > 0:
            self.force_right_escape -= 1
            return 0.9

        if angle < 0:
            self.left_escape_count += 1
            if self.left_escape_count >= 4:
                self.force_right_escape = 2
                self.left_escape_count = 0
        else:
            self.left_escape_count = 0
        return angle

    # ============================================================
    # MAX GAP 탐색 (수정 금지)
    # ============================================================
    def find_gap_max(self):
        if len(self.scan_ranges) == 0:
            return 0.0

        raw = np.array(self.scan_ranges)
        ranges = np.concatenate([raw[-60:], raw[:60]])
        ranges = np.where((ranges < 0.20) | np.isnan(ranges), 0.0, ranges)

        idx = np.argmax(ranges)
        max_dist = ranges[idx]

        if max_dist < (self.robot_width + 0.10):
            return 0.0

        angle_deg = idx - 60
        return angle_deg * np.pi / 180


if __name__ == "__main__":
    try:
        LineTracerWithObstacleAvoidance()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
