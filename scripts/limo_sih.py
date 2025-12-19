#!/usr/bin/env python3
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

        self.scan_ranges = []
        self.front = 999.0

        self.state = "LANE"
        self.escape_angle = 0.0
        self.state_start = rospy.Time.now().to_sec()

        self.left_escape_count = 0
        self.force_right_escape = 0

        self.robot_width = 0.13

        # ===== 라인트레이싱 파라미터 =====
        self.base_gain = 1.0 / 200.0
        self.corner_scale = 120.0
        self.max_steer = 0.9

        # ===== 라바콘 안정화(매끈하게) =====
        self.cone_span_ratio = 0.25 # 라바콘이 좌우로 충분히 퍼진 경우만 "사이"로 판단 (0.20~0.35)
        self.mid_filt = None
        self.mid_alpha = 0.75 # mid 필터 강도 (0.65~0.85)
        self.cone_last_w = 0.0
        self.cone_max_step = 0.10 # 라바콘 구간 각속도 변화 제한 (0.08~0.14)
        self.cone_w_limit = 0.70 # 라바콘 각속도 절대 제한

    # ============================================================
    # LIDAR
    # ============================================================
    def lidar_cb(self, scan):
        raw = np.array(scan.ranges)
        self.scan_ranges = raw

        front_zone = np.concatenate([raw[:10], raw[-10:]])
        cleaned = [d for d in front_zone if d > 0.20 and not np.isnan(d)]
        self.front = np.median(cleaned) if cleaned else 999.0

    # ============================================================
    # CAMERA
    # ============================================================
    def camera_cb(self, msg):
        twist = Twist()
        now = rospy.Time.now().to_sec()

        if self.state == "ESCAPE":
            self.escape_control()
            return

        if self.state == "BACK":
            self.back_control()
            return

        if self.state == "LANE":

            if self.front < 0.45:
                self.state = "BACK"
                self.state_start = now
                return

            frame = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
            h, w = frame.shape[:2]

            # ==========================
            # ROI 설정
            # ==========================
            roi_near = frame[int(h*0.55):h, :]
            hsv_near = cv2.cvtColor(roi_near, cv2.COLOR_BGR2HSV)

            # ==========================
            # 흰색 차선
            # ==========================
            lower_white = np.array([0, 0, 180])
            upper_white = np.array([180, 40, 255])
            mask_near = cv2.inRange(hsv_near, lower_white, upper_white)

            contours_near, _ = cv2.findContours(
                mask_near, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
            )

            # ==========================
            # 라바콘 (빨간색)
            # ==========================
            lower_r1 = np.array([0, 120, 80])
            upper_r1 = np.array([10, 255, 255])
            lower_r2 = np.array([170, 120, 80])
            upper_r2 = np.array([180, 255, 255])

            mask_r1 = cv2.inRange(hsv_near, lower_r1, upper_r1)
            mask_r2 = cv2.inRange(hsv_near, lower_r2, upper_r2)
            red_mask = cv2.bitwise_or(mask_r1, mask_r2)

            # 빨강 마스크 안정화(잡음/구멍 줄이기)
            kernel = np.ones((5, 5), np.uint8)
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

            red_contours, _ = cv2.findContours(
                red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            centers = []
            for cnt in red_contours:
                if cv2.contourArea(cnt) < 200:
                    continue
                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue
                centers.append(int(M["m10"] / M["m00"]))

            # ==========================
            # 라바콘 주행 (진입 안정 + 내부 매끄러움)
            # ==========================
            if len(centers) >= 1:
                centers.sort()

                # 라바콘 1개/한쪽 몰림이면 "사이 중앙"으로 급꺾지 말기
                use_gap_mid = False
                if len(centers) >= 2:
                    cone_span = centers[-1] - centers[0]
                    if cone_span > w * self.cone_span_ratio:
                        use_gap_mid = True

                if use_gap_mid:
                    mid = (centers[0] + centers[-1]) // 2
                else:
                    # 한쪽만 보이는 상황: 튀지 말고 직진/완만 유지용으로 화면 중앙 목표
                    # (원래 코드의 '직진' 성질을 유지하면서도 부드럽게)
                    mid = w // 2

                # mid 저역통과 필터 (툭툭 방지)
                if self.mid_filt is None:
                    self.mid_filt = float(mid)
                else:
                    self.mid_filt = self.mid_alpha * self.mid_filt + (1 - self.mid_alpha) * float(mid)
                mid_use = int(self.mid_filt)

                error = mid_use - (w // 2)
                target_w = float(np.clip(error / 180.0, -self.cone_w_limit, self.cone_w_limit))

                # 라바콘 구간 각속도 변화율 제한 (더 매끈하게)
                dw = target_w - self.cone_last_w
                dw = float(np.clip(dw, -self.cone_max_step, self.cone_max_step))
                w_cmd = self.cone_last_w + dw
                self.cone_last_w = w_cmd

                twist.linear.x = 0.21 if use_gap_mid else 0.16
                twist.angular.z = w_cmd
                self.pub.publish(twist)
                return

            # 라바콘이 안 보이면 필터/라바콘 조향 상태는 천천히 잊게 (과거 영향 최소화)
            self.mid_filt = None
            self.cone_last_w *= 0.7

            # ==========================
            # 라인 없음 → 직진 유지
            # ==========================
            if len(contours_near) == 0:
                twist.linear.x = 0.12
                twist.angular.z = 0.0
                self.pub.publish(twist)
                return

            # ==========================
            # 정상 라인트레이싱 (코너 조향 완화)
            # ==========================
            c = max(contours_near, key=cv2.contourArea)
            M = cv2.moments(c)

            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                error = cx - (w // 2)

                gain = self.base_gain / (1.0 + abs(error) / self.corner_scale)
                twist.linear.x = 0.22
                twist.angular.z = gain * error

                twist.angular.z = max(
                    min(twist.angular.z, self.max_steer),
                    -self.max_steer
                )

                self.pub.publish(twist)
                return

    # ============================================================
    # BACK MODE
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
    # ESCAPE MODE
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
    # ESCAPE 방향 보정
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
    # 최대 GAP 탐색
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
    LineTracerWithObstacleAvoidance()
    rospy.spin()
