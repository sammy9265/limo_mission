#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
import math
from sensor_msgs.msg import CompressedImage, LaserScan
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

class IntegratedLimoMission:
    def __init__(self):
        rospy.init_node("integrated_limo_mission", anonymous=True)
        
        # Publisher & Subscriber
        self.pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        rospy.Subscriber("/usb_cam/image_raw/compressed", CompressedImage, self.camera_cb)
        rospy.Subscriber("/scan", LaserScan, self.lidar_cb)

        self.bridge = CvBridge()

        # ==========================================================
        # [설정 A] 장애물 회피 & 라바콘 (첫 번째 코드 파라미터)
        # ==========================================================
        self.robot_width = 0.13
        self.scan_ranges = []
        self.front = 999.0
        
        self.state = "LANE"  # LANE, BACK, ESCAPE
        self.state_start = rospy.Time.now().to_sec()
        self.escape_angle = 0.0
        self.left_escape_count = 0
        self.force_right_escape = 0

        # 빨간색 HSV (라바콘용)
        self.lower_r1 = np.array([0, 120, 80])
        self.upper_r1 = np.array([10, 255, 255])
        self.lower_r2 = np.array([170, 120, 80])
        self.upper_r2 = np.array([180, 255, 255])

        # ==========================================================
        # [설정 B] 정교한 라인 트레이싱 (두 번째 코드 파라미터)
        # ==========================================================
        self.line_speed = 0.15        # 라인 주행 속도
        
        # 검은색 + 노란색 HSV (튜닝값 유지)
        self.black_lower = np.array([102, 0, 60])
        self.black_upper = np.array([164, 86, 136])
        self.black2_lower = np.array([126, 25, 45])
        self.black2_upper = np.array([167, 89, 108])
        self.black3_lower = np.array([125, 29, 26])
        self.black3_upper = np.array([171, 100, 78])
        
        self.yellow_lower = np.array([14, 17, 153])
        self.yellow_upper = np.array([35, 167, 255])
        
        # Bird's Eye View 파라미터
        self.margin_x = 150
        self.margin_y = 350
        
        # PID & Smoothing 제어 변수
        self.steer_weight = 2.0   
        self.steer_alpha = 0.35   
        self.steer_max = 1.20     
        self.steer_rate = 0.14    
        
        self.steer = 0.0
        self.steer_f = 0.0

        rospy.loginfo("===== Integrated Mission Node Started =====")

    # ============================================================
    # 1. LIDAR 처리 (장애물 감지 - 첫 번째 코드 로직)
    # ============================================================
    def lidar_cb(self, scan):
        raw = np.array(scan.ranges)
        self.scan_ranges = raw

        front_zone = np.concatenate([raw[:10], raw[-10:]])
        cleaned = [d for d in front_zone if d > 0.20 and not np.isnan(d)]
        self.front = np.median(cleaned) if cleaned else 999.0

    # ============================================================
    # 2. CAMERA 처리 (메인 판단 로직)
    # ============================================================
    def camera_cb(self, msg):
        twist = Twist()
        now = rospy.Time.now().to_sec()

        # [상태 1] 회피 기동 중이면 카메라 무시하고 회피 로직 수행
        if self.state == "ESCAPE":
            self.escape_control()
            return
        if self.state == "BACK":
            self.back_control()
            return

        # [상태 2] 주행 모드 (LANE)
        if self.state == "LANE":
            # 2-1. 장애물 충돌 위험 감지 (0.45m) -> 후진 모드로 전환
            if self.front < 0.45:
                rospy.logwarn("Obstacle Detected! -> BACK Mode")
                self.state = "BACK"
                self.state_start = now
                return

            try:
                # 이미지 변환
                frame = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
                h, w = frame.shape[:2]
                hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                # =========================================================
                # [우선순위 1] 빨간색 라바콘 감지 (미션 1~4)
                # =========================================================
                roi_near = frame[int(h*0.55):h, :]
                hsv_near = cv2.cvtColor(roi_near, cv2.COLOR_BGR2HSV)

                mask_r1 = cv2.inRange(hsv_near, self.lower_r1, self.upper_r1)
                mask_r2 = cv2.inRange(hsv_near, self.lower_r2, self.upper_r2)
                red_mask = cv2.bitwise_or(mask_r1, mask_r2)

                red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # 유효한 빨간색 컨투어 필터링
                valid_red = [cnt for cnt in red_contours if cv2.contourArea(cnt) > 200]

                if len(valid_red) >= 1:
                    # --- 빨간색이 보이면 첫 번째 코드의 주행 로직 사용 ---
                    centers = []
                    for cnt in valid_red:
                        M = cv2.moments(cnt)
                        if M["m00"] > 0:
                            centers.append(int(M["m10"] / M["m00"]))
                    
                    if len(centers) >= 2:
                        centers = sorted(centers)
                        mid = (centers[0] + centers[-1]) // 2
                    elif len(centers) == 1:
                        mid = centers[0]
                    else:
                        mid = w // 2

                    error = mid - (w // 2)
                    
                    twist.linear.x = 0.21
                    twist.angular.z = error / 180.0
                    self.pub.publish(twist)
                    
                    # 라바콘 모드 종료 후 라인 모드 진입 시 튐 방지용 동기화
                    self.steer = twist.angular.z
                    self.steer_f = twist.angular.z
                    return  # 빨간색 처리했으므로 여기서 끝

                # =========================================================
                # [우선순위 2] 빨간색 없음 -> BEV 라인 트레이싱 (미션 5~6)
                # =========================================================
                # 여기가 두 번째 코드 로직이 들어가는 부분입니다.
                
                # 색상 필터링 (Yellow + Black)
                yellow_filter = cv2.inRange(hsv_img, self.yellow_lower, self.yellow_upper)
                
                b1 = cv2.inRange(hsv_img, self.black_lower, self.black_upper)
                b2 = cv2.inRange(hsv_img, self.black2_lower, self.black2_upper)
                b3 = cv2.inRange(hsv_img, self.black3_lower, self.black3_upper)
                black_filter = cv2.bitwise_or(b1, cv2.bitwise_or(b2, b3))
                
                target_filter = cv2.bitwise_or(black_filter, yellow_filter)

                # BEV 변환 (두 번째 코드 설정값)
                src_pts = np.float32([
                    (30, h), 
                    (self.margin_x, self.margin_y), 
                    (w - self.margin_x, self.margin_y), 
                    (w - 30, h)
                ])
                dst_margin_x = 120
                dst_pts = np.float32([
                    (dst_margin_x, h), 
                    (dst_margin_x, 0), 
                    (w - dst_margin_x, 0), 
                    (w - dst_margin_x, h)
                ])
                matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
                warp_img = cv2.warpPerspective(target_filter, matrix, (w, h))

                # 모멘트 계산 및 PID 제어
                M = cv2.moments(warp_img)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    error = (w // 2) - cx
                    
                    # P Control
                    steer_raw = (error * math.pi / w) * self.steer_weight
                    steer_raw = float(np.clip(steer_raw, -self.steer_max, self.steer_max))

                    # Smoothing
                    self.steer_f = (1.0 - self.steer_alpha) * self.steer_f + self.steer_alpha * steer_raw
                    
                    # Rate Limiting
                    d = self.steer_f - self.steer
                    d = float(np.clip(d, -self.steer_rate, self.steer_rate))
                    self.steer = float(np.clip(self.steer + d, -self.steer_max, self.steer_max))
                else:
                    # 라인을 놓쳤을 때 (천천히 직진 혹은 약한 회전 유지)
                    self.steer = float(np.clip(self.steer, -0.45, 0.45))

                # 최종 주행 명령
                twist.linear.x = self.line_speed
                twist.angular.z = self.steer
                self.pub.publish(twist)

            except Exception as e:
                rospy.logerr(f"Error: {e}")

    # ============================================================
    # 3. 회피 기동 로직 (첫 번째 코드 그대로 사용)
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

    def escape_control(self):
        twist = Twist()
        now = rospy.Time.now().to_sec()

        if now - self.state_start < 1.0:
            twist.linear.x = 0.19
            twist.angular.z = self.escape_angle * 1.3
            self.pub.publish(twist)
        else:
            self.state = "LANE"
            # 라인 모드 복귀 시 PID 초기화
            self.steer = 0.0
            self.steer_f = 0.0

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
        IntegratedLimoMission()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
