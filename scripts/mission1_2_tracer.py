#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist

class LineTracerNode:
    def __init__(self):
        rospy.init_node('line_tracer_node', anonymous=False)
        rospy.loginfo("Line Tracer Node Started (Black Path Following)")

        # === [★ 튜닝 파라미터 영역 ★] ===
        # 1. 이미지 및 ROI 설정
        self.img_width = 320    # 연산 속도를 위한 해상도
        self.img_height = 240
        self.roi_ratio = 0.3    # 이미지 아래쪽 30% 영역만 분석
        
        # 2. 주행 속도 및 제어 게인
        self.linear_speed = 0.15 # m/s (직진 속도)
        self.kp_gain = 0.005     # P-제어 게인 (가장 중요!)
        
        # 3. 색상 임계값 (검은색 길 마스킹)
        # HSV 기준: 검은색은 Hue와 Saturation에 관계없이 Value(명도)가 낮음
        self.lower_black = np.array([0, 0, 0])     # H: 0~179, S: 0~255, V: 0~40 (어두운 영역)
        self.upper_black = np.array([179, 255, 40]) # V(명도)를 40 이하로 낮게 설정

        # ROS 통신
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        # LIMO 기본 카메라 토픽: '/camera/rgb/image_raw/compressed'
        self.sub = rospy.Subscriber('/camera/rgb/image_raw/compressed', 
                                    CompressedImage, self.img_callback, queue_size=1)
        
        self.center_x = self.img_width // 2 # 이미지 중앙값 (160)

    def img_callback(self, msg):
        try:
            # 1. 이미지 디코딩 및 리사이징 (경량화)
            np_arr = np.frombuffer(msg.data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (self.img_width, self.img_height))
            
            # 2. 전처리 (HSV 변환)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # 3. 검은색(길) 마스크 생성
            mask = cv2.inRange(hsv, self.lower_black, self.upper_black)
            
            # 4. 관심 영역(ROI) 설정 및 적용
            roi_h = int(self.img_height * self.roi_ratio)
            # 이미지 아래쪽(self.img_height - roi_h)부터 끝까지 분석
            mask_roi = mask[self.img_height - roi_h:, :]

            # 5. 무게중심(Centroid) 계산 (가장 빠른 방법)
            M = cv2.moments(mask_roi)
            
            twist = Twist()
            twist.linear.x = self.linear_speed

            if M['m00'] > 0:
                # 검은색 영역이 있을 경우
                cx = int(M['m10'] / M['m00'])
                
                # ROI가 아닌 전체 이미지 좌표계로 변환 (Optional, 편의상 사용)
                # cx = cx + 0 # ROI 시작점 보정 (여기서는 생략)
                
                # 6. 오차 계산 및 P-제어
                # 오차: 이미지 중앙(160) - 검은색 길의 중앙(cx)
                error = self.center_x - cx
                
                # 조향각 = 오차 * 게인
                twist.angular.z = error * self.kp_gain
                
                # 7. 안전장치: 조향각 제한 (-1.5 ~ 1.5 rad/s)
                twist.angular.z = max(min(twist.angular.z, 1.5), -1.5)
                
            else:
                # 검은색 길이 보이지 않을 경우 (선 이탈)
                rospy.logwarn_throttle(1.0, "Black line lost. Stopping.")
                twist.linear.x = 0.0 # 멈춤 (혹은 마지막 각도로 회전 유지)
                twist.angular.z = 0.0

            # 8. 명령 발행
            self.pub.publish(twist)

        except Exception as e:
            rospy.logwarn(f"Line Tracer Error: {e}")

if __name__ == '__main__':
    try:
        LineTracerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
