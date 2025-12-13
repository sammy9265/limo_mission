#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist

class Mission4ConeTracerLight:
    def __init__(self):
        rospy.init_node('mission4_camera_node', anonymous=False)
        
        # === [최적화 튜닝 파라미터] ===
        # 해상도를 160x120으로 낮춤 (속도 4배 향상)
        self.img_width = rospy.get_param("~img_width", 160)
        self.img_height = rospy.get_param("~img_height", 120)
        
        # ROI 비율 (화면 아래쪽 40%만 봄)
        self.roi_ratio = 0.4
        self.roi_h = int(self.img_height * self.roi_ratio)

        # 빨간색 감지 임계값 (해상도에 맞춰 자동 보정 필요할 수 있음)
        # 320x240일 때 500개였다면, 160x120에서는 125개 정도가 적당함
        default_thresh = 100
        self.red_detect_thresh = rospy.get_param("~red_detect_thresh", default_thresh)
        
        # 주행 설정
        self.speed = rospy.get_param("~speed", 0.15)
        self.turn_gain_lane = rospy.get_param("~turn_gain_lane", 0.005) # 반응성 약간 높임
        self.turn_gain_cone = rospy.get_param("~turn_gain_cone", 0.008)

        self.is_cone_mode = False 
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.sub = rospy.Subscriber('/camera/rgb/image_raw/compressed', 
                                    CompressedImage, self.img_callback, queue_size=1, buff_size=2**16)
        
        rospy.loginfo(f"Light Mode Started. Size: {self.img_width}x{self.img_height}, ROI Height: {self.roi_h}")

    def img_callback(self, msg):
        try:
            # 1. 디코딩 및 리사이징
            np_arr = np.frombuffer(msg.data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None: return
            
            # 리사이징 (여기서 이미 노이즈가 어느정도 잡힘)
            img = cv2.resize(img, (self.img_width, self.img_height))
            
            # 2. [최적화 핵심] ROI 먼저 자르기! (전체 변환 X, 필요한 부분만 변환 O)
            # 화면 아래쪽만 잘라냅니다.
            roi_img = img[self.img_height - self.roi_h:, :]
            
            # 3. 잘라낸 부분만 HSV 변환 (연산량 대폭 감소)
            hsv_roi = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)

            # === [모드 결정 로직] ===
            # 빨간색 마스크 (Red wraps around 180-0)
            # inRange 연산을 ROI에만 수행
            mask_red = cv2.inRange(hsv_roi, np.array([0, 100, 50]), np.array([10, 255, 255]))
            mask_red_2 = cv2.inRange(hsv_roi, np.array([170, 100, 50]), np.array([180, 255, 255]))
            mask_red = cv2.bitwise_or(mask_red, mask_red_2)
            
            red_pixel_count = cv2.countNonZero(mask_red)

            # 모드 전환 (한번 켜지면 계속 유지)
            if not self.is_cone_mode and red_pixel_count > self.red_detect_thresh:
                rospy.loginfo(f"Red Detected ({red_pixel_count} px). Mode: CONE")
                self.is_cone_mode = True

            # === [주행 제어] ===
            twist = Twist()
            twist.linear.x = self.speed

            if self.is_cone_mode:
                # [모드 2] 라바콘 반발력 주행
                # 이미 ROI로 잘려 있으므로 그대로 반으로 나눔
                half_w = self.img_width // 2
                left_mass = cv2.countNonZero(mask_red[:, :half_w])
                right_mass = cv2.countNonZero(mask_red[:, half_w:])
                
                # 오른쪽이 많으면 왼쪽으로 회전 (+)
                error = right_mass - left_mass
                twist.angular.z = error * self.turn_gain_cone
                
            else:
                # [모드 1] 흰색 차선 추종
                # 흰색 마스크 생성 (ROI 이미지 사용)
                # 조명에 따라 V값(180) 튜닝 필요
                mask_white = cv2.inRange(hsv_roi, np.array([0, 0, 160]), np.array([179, 50, 255]))

                M = cv2.moments(mask_white)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    # 화면 중앙(80) - 차선중심(cx)
                    error = (self.img_width / 2) - cx
                    twist.angular.z = error * self.turn_gain_lane
                else:
                    twist.angular.z = 0

            # 조향 제한
            twist.angular.z = max(min(twist.angular.z, 1.2), -1.2)
            self.pub.publish(twist)

        except Exception as e:
            rospy.logwarn_throttle(1, f"Img Error: {e}")

if __name__ == '__main__':
    try:
        Mission4ConeTracerLight()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
