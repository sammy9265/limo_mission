#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
import math
from sensor_msgs.msg import CompressedImage, LaserScan
from geometry_msgs.msg import Twist

class MissionIntegratedFinal:
    def __init__(self):
        rospy.init_node('mission_integrated_final', anonymous=False)
        
        # === [ROS 통신 설정] ===
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.sub_img = rospy.Subscriber('/camera/rgb/image_raw/compressed', 
                                        CompressedImage, self.img_callback, queue_size=1)
        self.sub_scan = rospy.Subscriber('/scan', LaserScan, self.scan_callback, queue_size=1)

        # === [상태 관리 변수] ===
        self.mode = "CAM"   
        self.start_time = rospy.Time.now().to_sec()
        self.force_cam_duration = 15.0 
        self.lidar_done = False  

        # ★ [추가된 변수] 복귀 딜레이 타이머
        self.clear_start_time = 0
        self.clear_wait_time = 2.0  # 2초 딜레이

        # === [1. 카메라 설정] ===
        self.img_width = 320
        self.img_height = 240
        self.cam_speed = 0.2          
        self.roi_ratio = 0.4
        self.roi_h = int(self.img_height * self.roi_ratio)
        
        self.lower_white = np.array([0, 0, 160]);     self.upper_white = np.array([179, 30, 255])
        self.lower_red1 = np.array([0, 100, 50]);     self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([170, 100, 50]);   self.upper_red2 = np.array([180, 255, 255])

        self.gain_white = 0.005
        self.gain_red = 0.008
        self.detect_thresh = 50

        # === [2. 라이다 설정] ===
        self.gap_speed = 0.22         
        self.gap_kp = 1.5             
        
        self.trigger_dist = 1.0       
        self.trigger_count = 0        
        self.trigger_limit = 5        

        self.safe_dist = 1.0
        self.view_limit_left = 15.0 
        self.view_limit_right = -80.0
        
        self.corner_safe_dist_default = 0.25 
        self.corner_safe_dist_narrow = 0.12 
        self.wall_ignore_y = 0.5
        
        self.recover_start_time = 0
        self.recover_duration = 1.5
        self.recover_dir = 0
        self.recover_rot_speed = 0.4

        rospy.loginfo("===== Mission Final: 2s Delay Return =====")
        rospy.loginfo(f"1. Force Camera: {self.force_cam_duration}s")
        rospy.loginfo("2. Lidar -> Cam: Wait 2.0s after clear")

    # ==========================================================
    # 1. 카메라 콜백
    # ==========================================================
    def img_callback(self, msg):
        if self.mode != "CAM" and self.mode != "FORCE_CAM": 
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
            rospy.logwarn_throttle(1, f"Img Error: {e}")

    # ==========================================================
    # 2. 라이다 콜백
    # ==========================================================
    def scan_callback(self, msg):
        # [A] 15초 강제 카메라
        if self.start_time == 0: self.start_time = rospy.Time.now().to_sec()
        elapsed = rospy.Time.now().to_sec() - self.start_time
        
        if elapsed < self.force_cam_duration:
            self.mode = "FORCE_CAM"
            return 

        # [B] 전처리
        ranges = np.array(msg.ranges)
        ranges[np.isinf(ranges)] = 10.0
        ranges[np.isnan(ranges)] = 10.0
        ranges[ranges < 0.1] = 10.0

        mid = len(ranges) // 2
        front_indices = ranges[mid-20 : mid+20]
        front_min = np.min(front_indices) if len(front_indices) > 0 else 10.0

        # [C] 후진 처리
        if self.mode == "RECOVER":
            if rospy.Time.now().to_sec() - self.recover_start_time > self.recover_duration:
                rospy.loginfo("[RECOVER] Done. Go Lidar.")
                self.mode = "LIDAR"
            else:
                twist = Twist()
                twist.linear.x = -0.2
                rot = self.recover_rot_speed
                twist.angular.z = rot if self.recover_dir == -1 else -rot
                self.pub.publish(twist)
            return

        # [D] 모드 전환 로직
        if self.mode == "CAM" or self.mode == "FORCE_CAM":
            if self.lidar_done: return 

            if front_min < self.trigger_dist:
                self.trigger_count += 1
                if self.trigger_count >= self.trigger_limit:
                    rospy.logwarn(f"!!! OBSTACLE ({front_min:.2f}m) -> LIDAR MODE START !!!")
                    self.mode = "LIDAR"
                    self.trigger_count = 0
            else:
                self.trigger_count = 0
            
            if self.mode == "CAM": return

        elif self.mode == "LIDAR":
            # ★ [수정] 2초 딜레이 로직 ★
            if front_min > (self.trigger_dist + 0.2): 
                # 처음으로 뚫렸으면 시간 측정 시작
                if self.clear_start_time == 0:
                    self.clear_start_time = rospy.Time.now().to_sec()
                    rospy.loginfo("Path Clear! Waiting 2s...")
                
                # 뚫린 상태로 2초가 지났는지 확인
                waited_time = rospy.Time.now().to_sec() - self.clear_start_time
                if waited_time >= self.clear_wait_time:
                    rospy.loginfo(f"!!! 2s PASSED ({front_min:.2f}m) -> SWITCH TO CAM !!!")
                    self.mode = "CAM"
                    self.lidar_done = True 
                    self.clear_start_time = 0
                    return
            else:
                # 다시 막히면 타이머 리셋
                self.clear_start_time = 0

        # [E] 라이다 주행 (Gap Follower)
        scan_data = []
        angle_min = msg.angle_min
        angle_inc = msg.angle_increment
        
        path_clear_dist = 10.0
        left_corner_min = 10.0
        right_corner_min = 10.0

        for i, r in enumerate(ranges):
            angle = angle_min + i * angle_inc
            deg = math.degrees(angle)
            
            if abs(deg) < 10:
                if r < path_clear_dist: path_clear_dist = r

            if 20 < deg < 60:
                if r < left_corner_min: left_corner_min = r
            elif -60 < deg < -20:
                if r < right_corner_min: right_corner_min = r
            
            if abs(deg) > 90: continue
            if self.view_limit_right < deg < self.view_limit_left:
                scan_data.append((angle, min(r, 3.0)))

        current_safe_limit = self.corner_safe_dist_default
        if path_clear_dist > 0.4:
            current_safe_limit = self.corner_safe_dist_narrow

        if left_corner_min < current_safe_limit:
            self.mode = "RECOVER"; self.recover_dir = 1; self.recover_start_time = rospy.Time.now().to_sec(); return
        elif right_corner_min < current_safe_limit:
            self.mode = "RECOVER"; self.recover_dir = -1; self.recover_start_time = rospy.Time.now().to_sec(); return

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
            if right_corner_min < 0.30: twist.angular.z += 0.3
            elif left_corner_min < 0.30: twist.angular.z -= 0.3
            twist.angular.z = np.clip(twist.angular.z, -1.2, 1.2)
        else:
            twist.linear.x = 0.0
            twist.angular.z = -0.6
            
        self.pub.publish(twist)

if __name__ == '__main__':
    try:
        MissionIntegratedFinal()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
