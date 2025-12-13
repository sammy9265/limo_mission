#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
import math
from sensor_msgs.msg import CompressedImage, LaserScan
from geometry_msgs.msg import Twist

class MissionMaster15s:
    def __init__(self):
        rospy.init_node('mission_master_15s_node', anonymous=False)
        
        # === [상태 관리] ===
        self.mode = "LINE"  # 초기 모드
        
        # ★ [시간 측정 변수 추가] ★
        self.start_time = rospy.Time.now().to_sec() # 시작 시간 기록
        self.force_line_duration = 15.0             # 15초 동안 강제 LINE 모드

        # === [1. 카메라 설정 (Mission4PriorityAvoid)] ===
        self.img_width = 320
        self.img_height = 240
        self.cam_speed = 0.2
        self.roi_ratio = 0.4
        self.roi_h = int(self.img_height * self.roi_ratio)
        
        # 색상 임계값
        self.lower_white = np.array([0, 0, 160]);     self.upper_white = np.array([179, 30, 255])
        self.lower_red1 = np.array([0, 100, 50]);     self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([170, 100, 50]);   self.upper_red2 = np.array([180, 255, 255])

        # 회피 게인
        self.gain_white = 0.005
        self.gain_red = 0.008
        self.detect_thresh = 50

        # === [2. 라이다 설정] ===
        self.gap_speed = 0.22
        self.gap_kp = 1.5
        
        self.trigger_dist = 1.0
        self.trigger_count = 0
        self.trigger_limit = 5
        self.clear_count = 0
        self.clear_limit = 10

        self.safe_dist = 1.0
        self.view_limit_left = 15.0 
        self.view_limit_right = -80.0

        # === [ROS 통신] ===
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.sub_img = rospy.Subscriber('/camera/rgb/image_raw/compressed', 
                                        CompressedImage, self.img_callback, queue_size=1)
        self.sub_scan = rospy.Subscriber('/scan', LaserScan, self.scan_callback, queue_size=1)

        rospy.loginfo("===== Mission Master Started (15s Force LINE) =====")

    # ==========================================================
    # 1. 카메라 콜백 (LINE 모드일 때만 동작)
    # ==========================================================
    def img_callback(self, msg):
        if self.mode != "LINE": 
            return

        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (self.img_width, self.img_height))
            
            blur = cv2.GaussianBlur(img, (5, 5), 0)
            hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
            roi = hsv[self.img_height - self.roi_h:, :]
            
            mask_r1 = cv2.inRange(roi, self.lower_red1, self.upper_red1)
            mask_r2 = cv2.inRange(roi, self.lower_red2, self.upper_red2)
            mask_red = cv2.bitwise_or(mask_r1, mask_r2)
            mask_white = cv2.inRange(roi, self.lower_white, self.upper_white)
            
            cnt_red = cv2.countNonZero(mask_red)
            cnt_white = cv2.countNonZero(mask_white)

            twist = Twist()
            twist.linear.x = self.cam_speed
            h, w = roi.shape[:2]
            center_x = w // 2

            if cnt_red > self.detect_thresh:
                left = cv2.countNonZero(mask_red[:, :center_x])
                right = cv2.countNonZero(mask_red[:, center_x:])
                twist.angular.z = (right - left) * self.gain_red

            elif cnt_white > self.detect_thresh:
                left = cv2.countNonZero(mask_white[:, :center_x])
                right = cv2.countNonZero(mask_white[:, center_x:])
                twist.angular.z = (right - left) * self.gain_white
            
            else:
                twist.angular.z = 0.0

            twist.angular.z = max(min(twist.angular.z, 1.5), -1.5)
            self.pub.publish(twist)

        except Exception as e:
            rospy.logwarn(f"Img Error: {e}")

    # ==========================================================
    # 2. 라이다 콜백 (모드 전환 + GAP 주행)
    # ==========================================================
    def scan_callback(self, msg):
        # ★ [핵심 추가] 15초 동안은 강제로 LINE 모드 유지 ★
        # 시작 시간이 0이면(초기화 안됨) 현재 시간으로 설정
        if self.start_time == 0: self.start_time = rospy.Time.now().to_sec()
        
        elapsed_time = rospy.Time.now().to_sec() - self.start_time
        
        if elapsed_time < self.force_line_duration:
            # 15초가 안 지났으면 -> 무조건 LINE 모드
            self.mode = "LINE"
            self.trigger_count = 0  # 카운터가 쌓이지 않게 초기화
            # rospy.loginfo_throttle(1, f"Force LINE Mode: {elapsed_time:.1f}s / 15.0s")
            return  # 라이다 로직은 실행하지 않고 종료 (카메라 콜백이 운전함)

        # --- 15초 이후: 원래 로직 실행 ---
        
        ranges = np.array(msg.ranges)
        ranges[np.isinf(ranges)] = 10.0
        ranges[np.isnan(ranges)] = 10.0
        ranges[ranges < 0.1] = 10.0

        mid = len(ranges) // 2
        front_indices = ranges[mid-20 : mid+20]
        front_min = np.min(front_indices) if len(front_indices) > 0 else 10.0

        # 모드 전환 로직
        if self.mode == "LINE":
            if front_min < self.trigger_dist:
                self.trigger_count += 1
                if self.trigger_count >= self.trigger_limit:
                    rospy.logwarn(f"!!! OBSTACLE ({front_min:.2f}m) -> SWITCH TO LIDAR !!!")
                    self.mode = "GAP"
                    self.trigger_count = 0
                    self.clear_count = 0
            else:
                self.trigger_count = 0

        elif self.mode == "GAP":
            if front_min > (self.trigger_dist + 0.2): 
                self.clear_count += 1
                if self.clear_count >= self.clear_limit:
                    rospy.loginfo(f"!!! PATH CLEAR ({front_min:.2f}m) -> BACK TO CAMERA !!!")
                    self.mode = "LINE"
                    self.clear_count = 0
                    self.trigger_count = 0
                    return
            else:
                self.clear_count = 0

            # GAP 주행 로직
            scan_data = []
            angle_min = msg.angle_min
            angle_inc = msg.angle_increment
            
            for i, r in enumerate(ranges):
                angle = angle_min + i * angle_inc
                deg = math.degrees(angle)
                if self.view_limit_right < deg < self.view_limit_left:
                    scan_data.append((angle, min(r, 3.0)))
            
            scan_data.sort(key=lambda x: x[0])
            max_gap_len = 0
            best_gap_center = 0.0
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
                            s, e = current_start, i-1
                            best_gap_center = scan_data[(s+e)//2][0]
                        current_start = -1
                        current_len = 0
            
            if current_start != -1 and current_len > max_gap_len:
                s, e = current_start, len(scan_data)-1
                best_gap_center = scan_data[(s+e)//2][0]
                max_gap_len = current_len

            twist = Twist()
            if max_gap_len > 5:
                twist.linear.x = self.gap_speed
                twist.angular.z = self.gap_kp * best_gap_center
                twist.angular.z = np.clip(twist.angular.z, -1.2, 1.2)
            else:
                twist.linear.x = 0.0
                twist.angular.z = -0.5

            self.pub.publish(twist)

if __name__ == '__main__':
    try:
        MissionMaster15s()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
