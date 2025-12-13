#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
import math
from sensor_msgs.msg import CompressedImage, LaserScan
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

class MissionFinalIntegrated:
    def __init__(self):
        rospy.init_node('mission_final_integrated', anonymous=False)
        
        # === [ROS 통신 설정] ===
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        
        # 카메라 토픽 (통합본 기준: usb_cam 사용)
        self.sub_img = rospy.Subscriber('/usb_cam/image_raw/compressed', 
                                        CompressedImage, self.img_callback, queue_size=1)
        # 라이다 토픽
        self.sub_scan = rospy.Subscriber('/scan', LaserScan, self.scan_callback, queue_size=1)
        
        self.bridge = CvBridge()

        # === [상태 관리 변수] ===
        self.mode = "CAM"   # 현재 모드 (CAM, LIDAR, RECOVER)
        self.start_time = rospy.Time.now().to_sec()  # 시작 시간
        self.force_cam_duration = 15.0               # 15초간 강제 카메라 모드

        # === [1. 카메라 설정 (Priority Avoidance)] ===
        self.img_width = 320
        self.img_height = 240
        self.cam_speed = 0.22         # 카메라 주행 속도
        self.roi_ratio = 0.4
        self.roi_h = int(self.img_height * self.roi_ratio)
        
        # 색상 임계값 (피해야 할 대상)
        self.lower_white = np.array([0, 0, 160]);     self.upper_white = np.array([179, 30, 255])
        self.lower_red1 = np.array([0, 100, 50]);     self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([170, 100, 50]);   self.upper_red2 = np.array([180, 255, 255])

        # 카메라 회피 게인
        self.gain_white = 0.005
        self.gain_red = 0.008
        self.detect_thresh = 50

        # === [2. 라이다 설정 (Gap Follower)] ===
        self.gap_speed = 0.22         # 라이다 주행 속도
        self.gap_kp = 1.5             # 조향 게인
        
        # 모드 전환 트리거
        self.trigger_dist = 1.0       # 이 거리 안에 장애물 있으면 라이다 모드
        self.trigger_count = 0        # 필터링 카운터
        self.trigger_limit = 5
        self.clear_count = 0
        self.clear_limit = 10

        # 라이다 주행 파라미터
        self.safe_dist = 1.0
        self.view_limit_left = 15.0 
        self.view_limit_right = -80.0
        
        # 충돌 감지 및 후진
        self.corner_safe_dist_default = 0.25 
        self.corner_safe_dist_narrow = 0.12 
        self.wall_ignore_y = 0.5
        
        self.recover_start_time = 0
        self.recover_duration = 1.5
        self.recover_dir = 0
        self.recover_rot_speed = 0.4

        rospy.loginfo("===== Final Integrated Mission Started =====")
        rospy.loginfo(f"1. Force Camera Mode for {self.force_cam_duration} seconds")
        rospy.loginfo("2. Then Dynamic Mode (Camera <-> Lidar)")

    # ==========================================================
    # 1. 카메라 콜백 (CAM 모드일 때 작동)
    #    기능: 흰색 선과 빨간색 라바콘을 '피하는' 로직
    # ==========================================================
    def img_callback(self, msg):
        # LIDAR나 RECOVER 모드일 때는 카메라는 잠시 쉰다
        if self.mode not in ["CAM", "FORCE_CAM"]: 
            return

        try:
            # 이미지 디코딩 및 리사이징
            np_arr = np.frombuffer(msg.data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (self.img_width, self.img_height))
            
            blur = cv2.GaussianBlur(img, (5, 5), 0)
            hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
            roi = hsv[self.img_height - self.roi_h:, :]
            
            # 마스크 생성 (빨강, 흰색)
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

            # [우선순위 회피 로직]
            # 왼쪽/오른쪽 무게를 비교해서 반대쪽으로 회전 (Repulsive Force)
            
            # 1순위: 빨간색 회피
            if cnt_red > self.detect_thresh:
                left = cv2.countNonZero(mask_red[:, :center_x])
                right = cv2.countNonZero(mask_red[:, center_x:])
                # 오른쪽이 많으면(양수) -> 왼쪽으로 회전(양수 angular) ...? 
                # 공식: error = right - left
                # 예: right(100) - left(0) = 100 -> +회전(좌회전). 
                # LIMO는 +가 좌회전이므로 장애물이 오른쪽에 있으면 왼쪽으로 감. (OK)
                twist.angular.z = (right - left) * self.gain_red

            # 2순위: 흰색 회피
            elif cnt_white > self.detect_thresh:
                left = cv2.countNonZero(mask_white[:, :center_x])
                right = cv2.countNonZero(mask_white[:, center_x:])
                twist.angular.z = (right - left) * self.gain_white
            
            # 3순위: 직진
            else:
                twist.angular.z = 0.0

            # 조향 제한
            twist.angular.z = max(min(twist.angular.z, 1.5), -1.5)
            self.pub.publish(twist)

        except Exception as e:
            rospy.logwarn(f"Img Error: {e}")

    # ==========================================================
    # 2. 라이다 콜백
    #    기능 1: 시간 체크 및 모드 전환
    #    기능 2: LIDAR 모드일 때 Gap Follower 주행
    #    기능 3: 충돌 감지 시 RECOVER 모드 실행
    # ==========================================================
    def scan_callback(self, msg):
        # --- [A] 15초 강제 카메라 모드 로직 ---
        elapsed = rospy.Time.now().to_sec() - self.start_time
        if elapsed < self.force_cam_duration:
            self.mode = "FORCE_CAM"
            # 15초 안 지났으면 라이다 로직은 무시하고 리턴
            return 

        # --- [B] 라이다 데이터 전처리 ---
        ranges = np.array(msg.ranges)
        ranges[np.isinf(ranges)] = 10.0
        ranges[np.isnan(ranges)] = 10.0
        ranges[ranges < 0.1] = 10.0

        # 정면 거리 (모드 전환용)
        mid = len(ranges) // 2
        front_indices = ranges[mid-20 : mid+20]
        front_min = np.min(front_indices) if len(front_indices) > 0 else 10.0

        # --- [C] 후진(Recover) 처리 ---
        if self.mode == "RECOVER":
            if rospy.Time.now().to_sec() - self.recover_start_time > self.recover_duration:
                rospy.loginfo("[RECOVER] Done. Gap Follow.")
                self.mode = "LIDAR"
            else:
                twist = Twist()
                twist.linear.x = -0.2
                rot = self.recover_rot_speed
                twist.angular.z = rot if self.recover_dir == -1 else -rot
                self.pub.publish(twist)
            return

        # --- [D] 모드 전환 로직 (CAM <-> LIDAR) ---
        if self.mode == "CAM" or self.mode == "FORCE_CAM":
            # 15초 지났고, CAM 모드인데 장애물이 가까우면 -> LIDAR 모드
            if front_min < self.trigger_dist:
                self.trigger_count += 1
                if self.trigger_count >= self.trigger_limit:
                    rospy.logwarn(f"!!! OBSTACLE ({front_min:.2f}m) -> LIDAR MODE !!!")
                    self.mode = "LIDAR"
                    self.trigger_count = 0
                    self.clear_count = 0
            else:
                self.trigger_count = 0
            
            # CAM 모드일 때는 여기서 끝 (주행은 img_callback이 함)
            if self.mode == "CAM": return

        elif self.mode == "LIDAR":
            # 장애물이 사라지면 -> CAM 모드
            if front_min > (self.trigger_dist + 0.2): 
                self.clear_count += 1
                if self.clear_count >= self.clear_limit:
                    rospy.loginfo(f"!!! PATH CLEAR ({front_min:.2f}m) -> CAM MODE !!!")
                    self.mode = "CAM"
                    self.clear_count = 0
                    self.trigger_count = 0
                    return
            else:
                self.clear_count = 0

        # --- [E] 라이다 주행 로직 (Gap Follower) ---
        # LIDAR 모드일 때만 실행됨
        
        # 1. 데이터 추출
        scan_data = []
        angle_min = msg.angle_min
        angle_inc = msg.angle_increment
        
        path_clear_dist = 10.0
        left_corner_min = 10.0
        right_corner_min = 10.0

        for i, r in enumerate(ranges):
            angle = angle_min + i * angle_inc
            deg = math.degrees(angle)
            
            # 전방 10도 거리
            if abs(deg) < 10:
                if r < path_clear_dist: path_clear_dist = r

            # 코너 감지
            if 20 < deg < 60:
                if r < left_corner_min: left_corner_min = r
            elif -60 < deg < -20:
                if r < right_corner_min: right_corner_min = r
            
            # Gap Finding용 데이터 (뒤쪽 제외, 시야각 제한)
            if abs(deg) > 90: continue
            if self.view_limit_right < deg < self.view_limit_left:
                scan_data.append((angle, min(r, 3.0)))

        # 2. 충돌 감지 -> RECOVER 전환
        current_safe_limit = self.corner_safe_dist_default
        if path_clear_dist > 0.4:
            current_safe_limit = self.corner_safe_dist_narrow

        if left_corner_min < current_safe_limit:
            rospy.logwarn("Left Hit -> Backup")
            self.mode = "RECOVER"
            self.recover_dir = 1
            self.recover_start_time = rospy.Time.now().to_sec()
            return
        elif right_corner_min < current_safe_limit:
            rospy.logwarn("Right Hit -> Backup")
            self.mode = "RECOVER"
            self.recover_dir = -1
            self.recover_start_time = rospy.Time.now().to_sec()
            return

        # 3. Gap Finding (빈 공간 찾기)
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

        # 4. 주행 명령 발행
        twist = Twist()
        if max_gap_len > 5:
            twist.linear.x = self.gap_speed
            twist.angular.z = self.gap_kp * best_gap_center
            
            # 코너 보정
            if right_corner_min < 0.30: twist.angular.z += 0.3
            elif left_corner_min < 0.30: twist.angular.z -= 0.3
            
            twist.angular.z = np.clip(twist.angular.z, -1.2, 1.2)
        else:
            twist.linear.x = 0.0
            twist.angular.z = -0.6 # 제자리 회전
            
        self.pub.publish(twist)

if __name__ == '__main__':
    try:
        MissionFinalIntegrated()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
