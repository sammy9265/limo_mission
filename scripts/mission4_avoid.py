#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist

class Mission4Avoid:
    def __init__(self):
        rospy.init_node('mission4_avoid_node', anonymous=False)
        
        # === [설정 파라미터] ===
        self.img_width = 320
        self.img_height = 240
        self.speed = 0.2          # 기본 주행 속도 (m/s)
        
        # 바닥 부분만 보기 위한 ROI 설정 (아래쪽 40%만 봄)
        self.roi_ratio = 0.4
        self.roi_h = int(self.img_height * self.roi_ratio)
        
        # 색상 임계값 (조명에 따라 튜닝 필요)
        # 흰색 차선
        self.lower_white = np.array([0, 0, 160])
        self.upper_white = np.array([179, 30, 255])
        
        # 빨간색 라바콘 (두 영역 합침)
        self.lower_red1 = np.array([0, 100, 50]);   self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([170, 100, 50]); self.upper_red2 = np.array([180, 255, 255])

        # 회피 민감도 (값이 클수록 확 꺾음)
        self.gain_avoid = 0.006 

        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.sub = rospy.Subscriber('/camera/rgb/image_raw/compressed', 
                                    CompressedImage, self.img_callback, queue_size=1)
        
        rospy.loginfo("Mission 4 Avoidance Mode Started: Avoiding RED & WHITE")

    def img_callback(self, msg):
        try:
            # 1. 이미지 처리
            np_arr = np.frombuffer(msg.data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (self.img_width, self.img_height))
            
            # 2. 블러 & HSV 변환
            blur = cv2.GaussianBlur(img, (5, 5), 0)
            hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
            
            # 3. ROI 설정 (바닥만 봄)
            # 화면 아래쪽만 잘라냅니다.
            roi = hsv[self.img_height - self.roi_h:, :]
            
            # --- 색상 마스크 생성 ---
            # 빨간색 (장애물)
            mask_r1 = cv2.inRange(roi, self.lower_red1, self.upper_red1)
            mask_r2 = cv2.inRange(roi, self.lower_red2, self.upper_red2)
            mask_red = cv2.bitwise_or(mask_r1, mask_r2)
            
            # 흰색 (차선 - 이것도 장애물로 취급!)
            mask_white = cv2.inRange(roi, self.lower_white, self.upper_white)
            
            # 두 장애물을 합침 (빨강 + 흰색 = "피해야 할 것들")
            mask_obstacle = cv2.bitwise_or(mask_red, mask_white)

            # === [회피 알고리즘: 반발력] ===
            # 화면을 왼쪽/오른쪽으로 나눔
            h, w = mask_obstacle.shape
            center_x = w // 2
            
            left_roi = mask_obstacle[:, :center_x]
            right_roi = mask_obstacle[:, center_x:]
            
            # 왼쪽/오른쪽에 "피해야 할 픽셀"이 얼마나 있는지 셈
            left_mass = cv2.countNonZero(left_roi)
            right_mass = cv2.countNonZero(right_roi)
            
            twist = Twist()
            twist.linear.x = self.speed

            # 회피 로직:
            # 왼쪽이 더 많으면 -> 오른쪽으로 가야 함 (angular.z < 0)
            # 오른쪽이 더 많으면 -> 왼쪽으로 가야 함 (angular.z > 0)
            # 공식: (오른쪽양 - 왼쪽양) * 게인
            # 예: 왼쪽(1000), 오른쪽(0) -> error = -1000 -> 우회전(음수) OK!
            
            error = right_mass - left_mass
            twist.angular.z = error * self.gain_avoid

            # 로그 출력 (디버깅용)
            if left_mass > 100 or right_mass > 100:
                rospy.loginfo(f"Avoid! L:{left_mass} R:{right_mass} -> Turn:{twist.angular.z:.2f}")

            # 조향 제한
            twist.angular.z = max(min(twist.angular.z, 1.5), -1.5)
            self.pub.publish(twist)

        except Exception as e:
            rospy.logwarn(f"Error: {e}")

if __name__ == '__main__':
    try:
        Mission4Avoid()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
