#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist

class Mission4PriorityAvoid:
    def __init__(self):
        rospy.init_node('mission4_priority_avoid_node', anonymous=False)
        
        # === [설정 파라미터] ===
        self.img_width = 320
        self.img_height = 240
        self.speed = 0.2          # 기본 주행 속도 (m/s)
        
        # 바닥 부분만 보기 위한 ROI 설정 (아래쪽 40%만 봄)
        self.roi_ratio = 0.4
        self.roi_h = int(self.img_height * self.roi_ratio)
        
        # === 색상 임계값 ===
        # 흰색 차선 (벽 2순위)
        self.lower_white = np.array([0, 0, 160])
        self.upper_white = np.array([179, 30, 255])
        
        # 빨간색 라바콘 (벽 1순위)
        self.lower_red1 = np.array([0, 100, 50]);   self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([170, 100, 50]); self.upper_red2 = np.array([180, 255, 255])

        # === 회피 게인 (Avoid Gain) ===
        # 값이 클수록 장애물 반대 방향으로 핸들을 확 꺾습니다.
        self.gain_white = 0.005  # 흰색 피할 때 민감도
        self.gain_red = 0.008    # 빨간색 피할 때 민감도 (더 무서워해야 함)

        # 감지 기준 (이 픽셀 수보다 적으면 장애물 없다고 판단)
        self.detect_thresh = 50

        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.sub = rospy.Subscriber('/camera/rgb/image_raw/compressed', 
                                    CompressedImage, self.img_callback, queue_size=1)
        
        rospy.loginfo("Priority Avoidance Mode: Red Wall > White Wall > Go Straight")

    def img_callback(self, msg):
        try:
            # 1. 이미지 처리
            np_arr = np.frombuffer(msg.data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (self.img_width, self.img_height))
            
            # 2. 전처리
            blur = cv2.GaussianBlur(img, (5, 5), 0)
            hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
            
            # 3. ROI 설정 (바닥만 봄)
            roi = hsv[self.img_height - self.roi_h:, :]
            
            # --- 색상 마스크 생성 ---
            # 빨간색
            mask_r1 = cv2.inRange(roi, self.lower_red1, self.upper_red1)
            mask_r2 = cv2.inRange(roi, self.lower_red2, self.upper_red2)
            mask_red = cv2.bitwise_or(mask_r1, mask_r2)
            
            # 흰색
            mask_white = cv2.inRange(roi, self.lower_white, self.upper_white)
            
            # 픽셀 수 확인
            cnt_red = cv2.countNonZero(mask_red)
            cnt_white = cv2.countNonZero(mask_white)

            twist = Twist()
            twist.linear.x = self.speed
            
            # 반발력 계산을 위한 화면 중앙 분할
            h, w = roi.shape[:2]
            center_x = w // 2

            # === [우선순위 회피 로직] ===
            
            # 1순위: 빨간색(라바콘)이 보이면 -> 빨간색으로부터 도망가기
            if cnt_red > self.detect_thresh:
                # 빨간색 마스크를 좌우로 나눔
                left_mass = cv2.countNonZero(mask_red[:, :center_x])
                right_mass = cv2.countNonZero(mask_red[:, center_x:])
                
                # [회피 공식] 오른쪽이 많으면 왼쪽으로(-), 왼쪽이 많으면 오른쪽으로(+)
                # error = right - left
                # 예: 왼쪽(100), 오른쪽(0) -> error = -100
                # angular.z = error * gain = -100 * 0.008 = -0.8 (우회전) -> 성공!
                
                error = right_mass - left_mass
                twist.angular.z = error * self.gain_red
                rospy.loginfo(f"!! AVOIDING RED !! L:{left_mass} R:{right_mass}")

            # 2순위: 빨간색 없고, 흰색(차선)이 보이면 -> 흰색으로부터 도망가기
            elif cnt_white > self.detect_thresh:
                # 흰색 마스크를 좌우로 나눔
                left_mass = cv2.countNonZero(mask_white[:, :center_x])
                right_mass = cv2.countNonZero(mask_white[:, center_x:])
                
                error = right_mass - left_mass
                twist.angular.z = error * self.gain_white
                rospy.loginfo(f"Avoiding White L:{left_mass} R:{right_mass}")
            
            # 3순위: 아무것도 안 보임 -> 직진 (길이 뚫림)
            else:
                twist.angular.z = 0.0
                # rospy.loginfo("Path Clear -> Go Straight")

            # 조향각 제한 (안전장치)
            twist.angular.z = max(min(twist.angular.z, 1.5), -1.5)
            self.pub.publish(twist)

        except Exception as e:
            rospy.logwarn(f"Error: {e}")

if __name__ == '__main__':
    try:
        Mission4PriorityAvoid()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
