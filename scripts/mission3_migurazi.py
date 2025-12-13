#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import math
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

class Mission3LidarNode:
    def __init__(self):
        rospy.init_node('mission3_obstacle_avoidance')
        
        # === [설정 파라미터] ===
        self.scan_fov = 120       # 인식 각도 (도)
        self.stop_dist = 0.5      # 장애물 감지 후 정지/회피 시작 거리 (m)
        self.critical_dist = 0.3  # 너무 가까워졌을 때 (후진 판단 거리) (m)
        self.min_gap_width = 0.10 # 주행 가능한 최소 통로 폭 (10cm)
        self.speed = 0.15         # 회피 주행 속도
        
        # ROS 통신
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.sub = rospy.Subscriber('/scan', LaserScan, self.lidar_cb, queue_size=1)
        
        self.is_avoiding = False  # 장애물 회피 모드 플래그

    def lidar_cb(self, msg):
        # 1. 라이다 데이터 전처리 (Inf, Nan 제거)
        ranges = np.array(msg.ranges)
        ranges = np.nan_to_num(ranges, nan=msg.range_max, posinf=msg.range_max)
        
        # 2. 정면 기준 120도(-60 ~ +60도) 데이터만 추출 및 120개로 압축
        # LIMO 라이다는 보통 0도가 정면이거나 뒤일 수 있으니, 정면 기준으로 인덱싱 필요
        # 여기서는 편의상 ranges의 중앙을 정면(0도)이라고 가정하고 슬라이싱합니다.
        # 실제 LIMO 라이다 인덱스에 맞춰 start/end 조정이 필요할 수 있습니다.
        
        mid_idx = len(ranges) // 2
        # 라이다 분해능(increment)을 고려해 60도에 해당하는 인덱스 개수 계산
        idx_range = int(math.radians(60) / msg.angle_increment)
        
        roi_ranges = ranges[mid_idx - idx_range : mid_idx + idx_range]
        
        # [경량화 핵심] 데이터를 120개(1도 단위)로 강제 리사이징 (평균값 사용)
        if len(roi_ranges) == 0: return
        
        # 원본 데이터를 120개 구간으로 나누어 평균을 냄 -> arr_120 (인덱스 0~119)
        # 인덱스 0: 우측 끝(-60도), 인덱스 60: 정면(0도), 인덱스 119: 좌측 끝(+60도)
        arr_120 = np.array([np.mean(chunk) for chunk in np.array_split(roi_ranges, 120)])

        # === [알고리즘 시작] ===
        
        min_d = np.min(arr_120)

        # 1. 평소(라인트레이싱 중)에는 50cm 이내 장애물 없으면 통과 (여기서는 cmd_vel 제어 안 함)
        # 만약 이 코드가 단독으로 돌면 직진 명령을 줘야 하지만, 통합시에는 패스
        if not self.is_avoiding and min_d > self.stop_dist:
            return 

        # 2. 장애물 감지! (50cm 이내) -> 회피 모드 진입
        self.is_avoiding = True
        twist = Twist()

        # [특수 상황] 거리가 30cm 이하로 가까워졌는데, 갈 수 있는 폭이 좁은 경우
        # 주행 가능 폭(Gap Width) 근사 계산: 거리 * 각도(라디안)
        # 가장 먼 곳(타겟)의 거리
        target_idx = np.argmax(arr_120)
        target_dist = arr_120[target_idx]
        
        # 타겟 주변의 연속된 빈 공간 개수를 대략적으로 파악 (단순화: 해당 거리 근처인 인덱스 개수)
        # 하지만 더 간단하게, 사용자가 말한대로 "주행할 수 있는 폭이 10cm도 안되면" 판별
        # 여기서는 타겟 지점의 거리를 기준으로 1도(약 0.017라디안) 당 폭을 계산
        # 폭 = 거리 * sin(개방각도). 
        # 간단히: 타겟 주변이 뚫려있는지 확인.
        
        # [후진 로직] 너무 가깝고(30cm), 정면 근처(인덱스 40~80)가 다 막혀있으면 후진
        front_min = np.min(arr_120[40:80])
        
        if front_min < self.critical_dist:
            # 주행 가능 공간 판단: 가장 먼 곳도 너무 가깝거나 막혀있다고 판단되면
            if target_dist < self.critical_dist + 0.1: # 여유분 10cm도 없으면
                rospy.logwarn("공간 부족! 후진 및 재탐색")
                self.perform_recovery()
                return

        # 3. 빈 공간(가장 먼 곳) 찾기 및 주행
        # 인덱스 0~119 중 값이 가장 큰 인덱스 찾기
        target_idx = np.argmax(arr_120) # 0~119 사이 값
        
        # 4. 조향각 계산 (사용자 요청 로직)
        # 인덱스 60이 0도. 
        # 인덱스 20 -> 20 - 60 = -40 (우측 40도 회전 필요, ROS 좌표계상 -가 우회전)
        # 인덱스 90 -> 90 - 60 = +30 (좌측 30도 회전 필요)
        target_angle_deg = target_idx - 60
        target_angle_rad = math.radians(target_angle_deg)
        
        rospy.loginfo(f"Target Idx: {target_idx}, Angle: {target_angle_deg} deg, Dist: {arr_120[target_idx]:.2f}m")

        # 주행 명령
        twist.linear.x = self.speed
        twist.angular.z = target_angle_rad * 1.5 # P게인 1.5배 (반응 빠르게)
        
        # 각도 제한 (안전장치)
        twist.angular.z = max(min(twist.angular.z, 1.5), -1.5)
        
        self.pub.publish(twist)

    def perform_recovery(self):
        """
        후진 로직: 멈춤 -> 왼쪽으로 핸들 틀고 후진 -> 멈춤 -> 다시 스캔
        """
        twist = Twist()
        
        # 1. 정지
        twist.linear.x = 0.0; twist.angular.z = 0.0
        self.pub.publish(twist)
        rospy.sleep(0.2)
        
        # 2. 왼쪽으로 핸들 최대한 틀어서 후진
        # 후진하면서 왼쪽으로 가려면, 뒷바퀴 기준 조향을 생각해야 함.
        # 단순히 뒤로 빼면서 엉덩이를 오른쪽으로 보내려면 -> angular.z는 양수(+)
        twist.linear.x = -0.15
        twist.angular.z = 1.0 # 왼쪽 회전(ROS 기준 +)
        
        # 1.5초간 후진
        start_time = rospy.Time.now()
        while (rospy.Time.now() - start_time).to_sec() < 1.5:
            self.pub.publish(twist)
            rospy.sleep(0.05)
            
        # 3. 정지 및 초기화
        twist.linear.x = 0.0; twist.angular.z = 0.0
        self.pub.publish(twist)
        rospy.sleep(0.5)
        
        # 상태 리셋 (다시 처음부터 먼 곳 찾도록)
        self.is_avoiding = True 

if __name__ == '__main__':
    try:
        Mission3LidarNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
