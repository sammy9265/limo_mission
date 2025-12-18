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

        rospy.Subscriber("/usb_cam/image_raw/compressed", CompressedImage, self.camera_cb)
        rospy.Subscriber("/scan", LaserScan, self.lidar_cb)

        self.bridge = CvBridge()

        # ==========================================
        # [설정] 검은색 라인트레이싱 파라미터 (Code 2)
        # ==========================================
        self.forward_speed = 0.12      # 검은색 라인 추종 시 전진 속도
        self.search_spin_speed = 0.25  # 라인 못 찾을 때 회전 속도
        self.k_angle = 0.010           # 조향 게인
        self.dark_min_pixels = 5       # 최소 검은색 픽셀 수
        self.dark_col_ratio = 0.3      # 임계값 비율

        # ==========================================
        # [설정] LiDAR 및 상태 변수 (Code 1 - 수정 금지)
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
    # LIDAR (Code 1 로직 유지)
    # ============================================================
    def lidar_cb(self, scan):
        raw = np.array(scan.ranges)
        self.scan_ranges = raw

        front_zone = np.concatenate([raw[:10], raw[-10:]])
        cleaned = [d for d in front_zone if d > 0.20 and not np.isnan(d)]
        self.front = np.median(cleaned) if cleaned else 999.0

    # ============================================================
    # CAMERA (빨간색 회피 -> 검은색 추종)
    # ============================================================
    def camera_cb(self, msg):
        twist = Twist()
        now = rospy.Time.now().to_sec()

        # 1. ESCAPE 모드
        if self.state == "ESCAPE":
            self.escape_control()
            return

        # 2. BACK 모드
        if self.state == "BACK":
            self.back_control()
            return

        # 3. LANE 모드
        if self.state == "LANE":

            # [LiDAR Check] 장애물 감지 시 BACK 모드 (최우선)
            if self.front < 0.45:
                self.state = "BACK"
                self.state_start = now
                return

            try:
                frame = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
                h, w = frame.shape[:2]
                center = w / 2.0

                # ROI 설정 (하단 45%)
                roi_h_start = int(h * 0.55)
                roi = frame[roi_h_start:h, :]
                
                # HSV 변환 (빨간색 감지용)
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

                # ================================================
                # [PRIORITY 1] 라바콘(빨간색) 검출 및 회피 (Code 1 로직)
                # ================================================
                lower_r1 = np.array([0, 120, 80])
                upper_r1 = np.array([10, 255, 255])
                lower_r2 = np.array([170, 120, 80])
                upper_r2 = np.array([180, 255, 255])

                mask_r1 = cv2.inRange(hsv, lower_r1, upper_r1)
                mask_r2 = cv2.inRange(hsv, lower_r2, upper_r2)
                red_mask = cv2.bitwise_or(mask_r1, mask_r2)

                red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if len(red_contours) >= 1:
                    centers = []
                    for cnt in red_contours:
                        if cv2.contourArea(cnt) < 200: continue
                        M = cv2.moments(cnt)
                        if M["m00"] == 0: continue
                        centers.append(int(M["m10"] / M["m00"]))

                    if len(centers) > 0:
                        if len(centers) >= 2:
                            centers = sorted(centers)
                            mid = (centers[0] + centers[-1]) // 2
                        else:
                            mid = int(centers[0])

                        error = mid - (w // 2)
                        twist.linear.x = 0.13
                        twist.angular.z = error / 180.0
                        self.pub.publish(twist)
                        return  # 빨간색이 보이면 여기서 함수 종료 (검은색 로직 실행 X)

                # ================================================
                # [PRIORITY 2] 검은색 트랙 추종 (Code 2 로직)
                # ================================================
                
                # 그레이스케일 변환 및 블러
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (5, 5), 0)

                # Otsu 이진화 (검은색 트랙 강조: THRESH_BINARY_INV)
                # 검은색 부분이 255(흰색)로 변하고, 나머지는 0이 됨
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

                # 노이즈 제거
                kernel = np.ones((3, 3), np.uint8)
                binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
                binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

                # 열별 픽셀 수 계산 (Column Sum)
                mask_black = (binary > 0)
                col_sum = np.sum(mask_black, axis=0)
                max_val = int(np.max(col_sum)) if col_sum.size > 0 else 0

                # 1. 검은색 라인이 너무 적으면 -> 못 찾음 -> 제자리 회전
                if max_val < self.dark_min_pixels:
                    twist.linear.x = 0.0
                    twist.angular.z = self.search_spin_speed
                    self.pub.publish(twist)
                    return

                # 2. 유효한 열 후보 추출
                threshold_val = max(self.dark_min_pixels, int(max_val * self.dark_col_ratio))
                candidates = np.where(col_sum >= threshold_val)[0]

                if candidates.size == 0:
                    twist.linear.x = 0.0
                    twist.angular.z = self.search_spin_speed
                    self.pub.publish(twist)
                    return

                # 3. 무게중심(Center of Mass) 계산
                # x 좌표들의 가중 평균을 구함 (검은색이 많은 곳이 중심)
                numerator = np.sum(candidates * col_sum[candidates])
                denominator = np.sum(col_sum[candidates])
                track_center_x = float(numerator / denominator)

                # 4. 조향각 계산
                # Code 2 로직: offset = track_center_x - center
                # ang = -self.k_angle * offset (왼쪽+, 오른쪽-)
                offset = track_center_x - center 
                ang = -self.k_angle * offset
                
                # 조향값 제한 (-0.8 ~ 0.8)
                ang = max(min(ang, 0.8), -0.8)

                # 최종 주행 명령 발행
                twist.linear.x = self.forward_speed
                twist.angular.z = ang
                self.pub.publish(twist)

            except Exception as e:
                rospy.logerr(f"Image Processing Error: {e}")

    # ============================================================
    # BACK MODE (Code 1 - 수정 금지)
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
    # ESCAPE MODE (Code 1 - 수정 금지)
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
    # ESCAPE 방향 로직 (Code 1 - 수정 금지)
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
    # MAX GAP 탐색 (Code 1 - 수정 금지)
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
