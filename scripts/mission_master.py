#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
import math
from sensor_msgs.msg import CompressedImage, LaserScan
from geometry_msgs.msg import Twist

class MissionMaster:
    def __init__(self):
        rospy.init_node('mission_master_node', anonymous=False)
        
        # === [상태 관리] ===
        self.mode = "LINE"  # 초기 모드: 카메라 주행 (LINE) / 라이다 주행 (GAP)
        
        # === [1. 카메라 설정 (Mission4PriorityAvoid)] ===
        self.img_width = 320
        self.img_height = 240
        self.cam_speed = 0.2          # 카메라 주행 속도
        self.roi_ratio = 0.4
        self.roi_h = int(self.img_height * self.roi_ratio)
        
        # 색상 임계값 (흰색/빨간색 회피용)
        self.lower_white = np.array([0, 0, 160]);     self.upper_white = np.array([179, 30, 255])
        self.lower_red1 = np.array([0, 100, 50]);     self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([170, 100, 50]);   self.upper_red2 = np.array([180, 255, 255])

        # 카메라 회피 게인
        self.gain_white = 0.005
        self.gain_red = 0.008
        self.detect_thresh = 50

        # === [2. 라이다 설정 (LimoMissionFinalFiltered)] ===
        self.gap_speed = 0.22         # 라이다 주행 속도
        self.gap_kp = 1.5             # 조향 게인
        
        # 모드 전환 트리거 설정
        self.trigger_dist = 1.0       # 이 거리 안에 장애물이 있으면 라이다 모드 ON
        self.trigger_count = 0        # 장애물 감지 카운터
        self.trigger_limit = 5        # 5번 연속 감지시 전환 (노이즈 방지)
        
        self.clear_count = 0          # 장애물 사라짐 카운터
        self.clear_limit = 10         # 10번 연속 없으면 다시 카메라 모드 복귀

        # 안전 거리 및 시야각
        self.safe_dist = 1.0
        self.view_limit_left = 15.0 
        self.view_limit_right = -80.0
        self.corner_safe_dist = 0.25

        # === [ROS 통신] ===
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.sub_img = rospy.Subscriber('/camera/rgb/image_raw/compressed', 
                                        CompressedImage, self.img_callback, queue_size=1)
        self.sub_scan = rospy.Subscriber('/scan', LaserScan, self.scan_callback, queue_size=1)

        rospy.loginfo("===== Mission Master Started =====")
        rospy.loginfo("Mode: LINE (Camera Avoid) <--> GAP (Lidar Avoid)")

    # ==========================================================
    # 1. 카메라 콜백 (LINE 모드일 때만 동작)
    # ==========================================================
    def img_callback(self, msg):
        # 라이다 모드(GAP)일 때는 카메라는 잠시 끕니다.
        if self.mode != "LINE": 
            return

        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (self.img_width, self.img_height))
            
            blur = cv2.GaussianBlur(img, (5, 5), 0)
            hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
            roi = hsv[self.img_height - self.roi_h:, :]
            
            # 마스크 생성
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
            if cnt_red > self.detect_thresh:
                left = cv2.countNonZero(mask_red[:, :center_x])
                right = cv2.countNonZero(mask_red[:, center_x:])
                twist.angular.z = (right - left) * self.gain_red
                # rospy.loginfo_throttle(1, "Camera: Avoiding RED")

            elif cnt_white > self.detect_thresh:
                left = cv2.countNonZero(mask_white[:, :center_x])
                right = cv2.countNonZero(mask_white[:, center_x:])
                twist.angular.z = (right - left) * self.gain_white
                # rospy.loginfo_throttle(1, "Camera: Avoiding WHITE")
            
            else:
                twist.angular.z = 0.0

            twist.angular.z = max(min(twist.angular.z, 1.5), -1.5)
            self.pub.publish(twist)

        except Exception as e:
            rospy.logwarn(f"Img Error: {e}")

    # ==========================================================
    # 2. 라이다 콜백 (모드 전환 판단 + GAP 모드 주행)
    # ==========================================================
    def scan_callback(self, msg):
        ranges = np.array(msg.ranges)
        # 전처리: Inf, 0 제거
        ranges[np.isinf(ranges)] = 10.0
        ranges[np.isnan(ranges)] = 10.0
        ranges[ranges < 0.1] = 10.0

        # 정면(0도 기준 +/- 10도) 거리 확인 -> 모드 전환용
        # LIMO 라이다 인덱스 계산 (msg.angle_min 등 고려 필요하지만, 간단히 중앙 인덱스 사용)
        mid = len(ranges) // 2
        front_indices = ranges[mid-20 : mid+20] # 대략 정면
        front_min = np.min(front_indices) if len(front_indices) > 0 else 10.0

        # === [모드 전환 로직] ===
        if self.mode == "LINE":
            # 장애물이 나타났다! (1.0m 이내)
            if front_min < self.trigger_dist:
                self.trigger_count += 1
                if self.trigger_count >= self.trigger_limit:
                    rospy.logwarn(f"!!! OBSTACLE ({front_min:.2f}m) -> SWITCH TO LIDAR !!!")
                    self.mode = "GAP"
                    self.trigger_count = 0
                    self.clear_count = 0
            else:
                self.trigger_count = 0 # 노이즈 리셋

        elif self.mode == "GAP":
            # 장애물이 사라졌다! (1.2m 이상 뻥 뚫림) -> 다시 카메라로
            if front_min > (self.trigger_dist + 0.2): 
                self.clear_count += 1
                if self.clear_count >= self.clear_limit:
                    rospy.loginfo(f"!!! PATH CLEAR ({front_min:.2f}m) -> BACK TO CAMERA !!!")
                    self.mode = "LINE"
                    self.clear_count = 0
                    self.trigger_count = 0
                    return # 이번 턴은 종료
            else:
                self.clear_count = 0

            # === [GAP 모드 주행 로직] ===
            # 여기서부터는 라이다로 주행 명령을 내립니다.
            
            # 유효한 데이터 추출 (각도, 거리)
            scan_data = []
            angle_min = msg.angle_min
            angle_inc = msg.angle_increment
            
            for i, r in enumerate(ranges):
                angle = angle_min + i * angle_inc
                deg = math.degrees(angle)
                
                # 뒤쪽은 무시, 필요한 시야각만 사용
                if self.view_limit_right < deg < self.view_limit_left:
                    scan_data.append((angle, min(r, 3.0)))
            
            # 갭(Gap) 찾기 알고리즘
            scan_data.sort(key=lambda x: x[0]) # 각도순 정렬
            max_gap_len = 0
            best_gap_center = 0.0
            current_start = -1
            current_len = 0

            for i in range(len(scan_data)):
                _, dist = scan_data[i]
                if dist > self.safe_dist: # 안전거리보다 멀면 '빈 공간'
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
            
            # 마지막 구간 체크
            if current_start != -1 and current_len > max_gap_len:
                s, e = current_start, len(scan_data)-1
                best_gap_center = scan_data[(s+e)//2][0]
                max_gap_len = current_len

            # 주행 명령 생성
            twist = Twist()
            if max_gap_len > 5: # 갈 수 있는 틈이 충분하면
                twist.linear.x = self.gap_speed
                twist.angular.z = self.gap_kp * best_gap_center
                twist.angular.z = np.clip(twist.angular.z, -1.2, 1.2)
            else:
                # 갇혔을 때 (제자리 회전 혹은 후진)
                twist.linear.x = 0.0
                twist.angular.z = -0.5

            self.pub.publish(twist)

if __name__ == '__main__':
    try:
        MissionMaster()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
