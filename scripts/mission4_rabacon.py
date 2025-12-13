#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist

class Mission4ConeTracer:
    def __init__(self):
        rospy.init_node('mission4_camera_node', anonymous=False)
        
        # === [설정 파라미터] ===
        self.img_width = 320   # 연산 속도를 위해 해상도 축소
        self.img_height = 240
        self.red_detect_thresh = 500  # 모드 전환을 위한 빨간색 픽셀 최소 개수
        
        # 주행 속도 및 게인 값 (튜닝 필요)
        self.speed = 0.15      # 기본 주행 속도 (m/s)
        self.turn_gain_lane = 0.004  # 흰색 차선 주행 조향 게인
        self.turn_gain_cone = 0.006  # 라바콘 주행 조향 게인 (반응이 더 빨라야 함)

        # 상태 변수 (False: 흰색 차선 모드, True: 라바콘 모드)
        self.is_cone_mode = False 

        # ROS 통신
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.sub = rospy.Subscriber('/camera/rgb/image_raw/compressed', 
                                    CompressedImage, self.img_callback, queue_size=1)

    def img_callback(self, msg):
        try:
            # 1. 이미지 디코딩 및 리사이징 (경량화)
            np_arr = np.frombuffer(msg.data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (self.img_width, self.img_height))
            
            # 2. 전처리 (Blur & HSV 변환)
            img_blur = cv2.GaussianBlur(img, (5, 5), 0)
            hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)

            # === [모드 결정 로직] ===
            # 빨간색 마스크 생성 (라바콘 감지)
            # 빨간색은 Hue 값이 0근처와 180근처 양쪽에 분포함
            lower_red1 = np.array([0, 100, 50]); upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 100, 50]); upper_red2 = np.array([180, 255, 255])
            mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask_red = cv2.bitwise_or(mask_red1, mask_red2)
            
            # 관심 영역(ROI) 설정 - 바닥 부분만 봄
            roi_h = int(self.img_height * 0.4)
            mask_red_roi = mask_red[self.img_height - roi_h:, :]

            # 빨간색 픽셀 수 확인
            red_pixel_count = cv2.countNonZero(mask_red_roi)

            # 모드 전환: 빨간색이 일정 이상 보이면 라바콘 모드로 고정 (한번 전환되면 돌아오지 않음)
            if not self.is_cone_mode and red_pixel_count > self.red_detect_thresh:
                rospy.loginfo("!! RED CONE DETECTED !! Switching to Cone Mode")
                self.is_cone_mode = True

            # === [주행 제어] ===
            twist = Twist()
            twist.linear.x = self.speed

            if self.is_cone_mode:
                # [모드 2] 라바콘 사이 주행 (반발력 알고리즘)
                # 화면을 좌우로 나누어 빨간색 양을 비교합니다.
                # 왼쪽 빨간색이 많으면 -> 오른쪽으로 회전 (오차: 음수)
                # 오른쪽 빨간색이 많으면 -> 왼쪽으로 회전 (오차: 양수)
                
                half_width = self.img_width // 2
                left_roi = mask_red_roi[:, :half_width]
                right_roi = mask_red_roi[:, half_width:]
                
                left_mass = cv2.countNonZero(left_roi)
                right_mass = cv2.countNonZero(right_roi)

                # 오른쪽이 많으면 왼쪽으로 틀어야 함 (+), 왼쪽이 많으면 오른쪽으로 (-)
                error = right_mass - left_mass
                twist.angular.z = error * self.turn_gain_cone
                
            else:
                # [모드 1] 흰색 차선 추종 (Line Tracing)
                # 흰색 마스크 생성 (HSV 기준, 조명에 따라 S, V 튜닝 필수)
                # 예: S(채도)가 낮고 V(명도)가 높은 색
                lower_white = np.array([0, 0, 180]) 
                upper_white = np.array([179, 40, 255])
                mask_white = cv2.inRange(hsv, lower_white, upper_white)
                
                # ROI 설정 (차선은 아래쪽만 봄)
                mask_white_roi = mask_white[self.img_height - roi_h:, :]

                # 무게중심(Moments) 계산
                M = cv2.moments(mask_white_roi)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    # 화면 중앙(160)과 차선 중심(cx)의 차이
                    error = self.img_width/2 - cx
                    twist.angular.z = error * self.turn_gain_lane
                else:
                    # 차선이 안 보이면 직진 혹은 회전 (상황에 맞게 수정)
                    twist.angular.z = 0

            # 명령 발행
            # 안전장치: 조향각 제한 (-1.0 ~ 1.0 rad/s)
            twist.angular.z = max(min(twist.angular.z, 1.0), -1.0)
            self.pub.publish(twist)

        except Exception as e:
            rospy.logwarn(f"Error: {e}")

if __name__ == '__main__':
    try:
        node = Mission4ConeTracer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
