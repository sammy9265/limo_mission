#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist

class Mission4Priority:
    def __init__(self):
        rospy.init_node('mission4_priority_node', anonymous=False)
        
        # === [설정 파라미터] ===
        self.img_width = 320
        self.img_height = 240
        self.speed = 0.2          # 기본 주행 속도 (m/s)
        
        # 바닥 부분만 보기 위한 ROI 설정 (아래쪽 40%만 봄)
        self.roi_ratio = 0.4
        self.roi_h = int(self.img_height * self.roi_ratio)
        
        # === 색상 임계값 ===
        # 흰색 차선 (HSV)
        self.lower_white = np.array([0, 0, 160])
        self.upper_white = np.array([179, 30, 255])
        
        # 빨간색 라바콘 (HSV - 0도와 180도 근처 두 영역 합침)
        self.lower_red1 = np.array([0, 100, 50]);   self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([170, 100, 50]); self.upper_red2 = np.array([180, 255, 255])

        # === 조향 게인 (튜닝 필요) ===
        # 값이 클수록 핸들을 확 꺾습니다.
        self.gain_white = 0.005  # 흰색 따라갈 때 민감도
        self.gain_red = 0.007    # 빨간색 따라갈 때 민감도 (더 강하게 반응 추천)

        # 감지 기준 픽셀 수 (이 값보다 커야 "보인다"고 판단)
        self.detect_thresh = 50

        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.sub = rospy.Subscriber('/camera/rgb/image_raw/compressed', 
                                    CompressedImage, self.img_callback, queue_size=1)
        
        rospy.loginfo("Mission 4 Priority Mode Started: White Default -> Red Priority")

    def img_callback(self, msg):
        try:
            # 1. 이미지 처리 및 리사이징
            np_arr = np.frombuffer(msg.data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (self.img_width, self.img_height))
            
            # 2. 블러 & HSV 변환
            blur = cv2.GaussianBlur(img, (5, 5), 0)
            hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
            
            # 3. ROI 설정 (바닥만 봄)
            roi = hsv[self.img_height - self.roi_h:, :]
            
            # --- 색상 마스크 생성 ---
            # 빨간색 (우선순위 높음)
            mask_r1 = cv2.inRange(roi, self.lower_red1, self.upper_red1)
            mask_r2 = cv2.inRange(roi, self.lower_red2, self.upper_red2)
            mask_red = cv2.bitwise_or(mask_r1, mask_r2)
            
            # 흰색 (우선순위 낮음)
            mask_white = cv2.inRange(roi, self.lower_white, self.upper_white)
            
            # 픽셀 수 확인 (얼마나 많이 보이는지)
            cnt_red = cv2.countNonZero(mask_red)
            cnt_white = cv2.countNonZero(mask_white)

            twist = Twist()
            twist.linear.x = self.speed
            center_x = self.img_width // 2  # 화면 중앙 좌표 (160)

            # === [우선순위 로직] ===
            
            # 1순위: 빨간색이 감지되면 -> 빨간색 라인트레이싱 (흰색 무시)
            if cnt_red > self.detect_thresh:
                # 빨간색 덩어리의 무게중심 계산
                M = cv2.moments(mask_red)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    # 화면 중앙과 빨간색 중심의 차이 (Error)
                    # 빨간색이 화면 중앙에 오도록 조향
                    error = center_x - cx  # (양수: 빨강이 왼쪽에 있음 -> 왼쪽 회전 필요)
                    twist.angular.z = error * self.gain_red
                    rospy.loginfo(f"!! RED TRACING !! (px: {cnt_red})")
                else:
                    twist.angular.z = 0

            # 2순위: 빨간색이 없고, 흰색이 감지되면 -> 흰색 라인트레이싱
            elif cnt_white > self.detect_thresh:
                # 흰색 덩어리의 무게중심 계산
                M = cv2.moments(mask_white)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    # 화면 중앙과 흰색 중심의 차이 (Error)
                    error = center_x - cx
                    twist.angular.z = error * self.gain_white
                    rospy.loginfo(f"White Tracing (px: {cnt_white})")
                else:
                    twist.angular.z = 0
            
            # 3순위: 아무것도 안 보임 -> 정지 (안전)
            else:
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                # rospy.loginfo("No Target Found")

            # 조향각 제한 (안전장치 -1.5 ~ 1.5 rad/s)
            twist.angular.z = max(min(twist.angular.z, 1.5), -1.5)
            self.pub.publish(twist)

        except Exception as e:
            rospy.logwarn(f"Error: {e}")

if __name__ == '__main__':
    try:
        Mission4Priority()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
