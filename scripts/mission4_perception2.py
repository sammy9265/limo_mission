#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32
import numpy as np

"""
미션 4: 라바콘 주행 (Perception - LiDAR) [VFH 알고리즘 적용]
- 규정: "라바콘으로 제시된 길을 충돌 없이 통과" (LiDAR 미션)
- 역할: VFH 알고리즘을 사용하여 라바콘 사이의 '안전한 길'을 찾음.
- 구독: /scan
- 발행: /mission4/target_angle (Float32, 단위: 도)
"""

class Mission4PerceptionNode:
    def __init__(self):
        rospy.init_node("mission4_perception_node")
        rospy.loginfo("LiDAR Perception Node (Mission 4: Rubber Cone) started")

        # === [미션 4 핵심 튜닝] ===
        # 1. 스캔 범위: 라바콘 코스는 곡선이므로 120~150도 정도면 충분함
        self.scan_angle_deg = 150.0 
        self.num_sectors = 36     # 150도를 36개 구역으로 나눔
        
        # 2. 장애물 인식 거리: 라바콘은 작으므로 너무 멀리 보면 놓칠 수 있음
        # (코스 폭을 고려하여 1.5m~2.0m 설정)
        self.obstacle_threshold_m = 1.8 
        
        # 3. 안전 반경 (버블): 라바콘과 충돌하지 않기 위한 여유 거리
        # 로봇 폭(0.2m) + 여유. 0.35m 정도 추천
        self.robot_width_m = 0.35 
        
        # === ROS IO ===
        self.scan_sub = rospy.Subscriber("/scan", LaserScan, self.scan_callback, queue_size=1)
        self.angle_pub = rospy.Publisher("/mission4/target_angle", Float32, queue_size=1)

        self.angles = None # 라이다 각도 배열
        self.sector_size_rad = np.deg2rad(self.scan_angle_deg) / self.num_sectors

    def init_angles(self, msg):
        """ 라이다 각도 배열 미리 계산 """
        self.angles = msg.angle_min + np.arange(len(msg.ranges)) * msg.angle_increment
        half_angle_rad = np.deg2rad(self.scan_angle_deg / 2.0)
        center_idx = len(msg.ranges) // 2
        scan_width_idx = int(half_angle_rad / msg.angle_increment)
        self.start_idx = max(0, center_idx - scan_width_idx)
        self.end_idx = min(len(msg.ranges), center_idx + scan_width_idx)
        rospy.loginfo(f"M4 VFH initialized. Scan range: {self.scan_angle_deg} deg.")

    def scan_callback(self, msg):
        if self.angles is None:
            self.init_angles(msg)
            return

        try:
            ranges = np.array(msg.ranges)
            ranges[np.isinf(ranges)] = msg.range_max
            ranges[np.isnan(ranges)] = 0.0
            ranges[ranges == 0.0] = msg.range_max

            # 1. 설정한 각도 범위만큼 데이터 자르기
            scan_ranges = ranges[self.start_idx:self.end_idx]
            scan_angles = self.angles[self.start_idx:self.end_idx]

            # 2. 히스토그램 생성 (장애물 분포 파악)
            histogram = np.full(self.num_sectors, msg.range_max)
            
            sector_indices = ((scan_angles + np.deg2rad(self.scan_angle_deg/2)) / self.sector_size_rad).astype(int)
            sector_indices = np.clip(sector_indices, 0, self.num_sectors - 1)

            for i in range(self.num_sectors):
                sector_points = scan_ranges[sector_indices == i]
                if len(sector_points) > 0:
                    histogram[i] = np.min(sector_points)

            # 3. 장애물 확장 (충돌 방지 버블)
            min_dist = np.min(histogram)
            if min_dist < 0.1: min_dist = 0.1
            
            bubble_angle_rad = np.arctan(self.robot_width_m / min_dist)
            bubble_radius_idx = int(bubble_angle_rad / self.sector_size_rad)
            
            inflated_histogram = np.copy(histogram)
            for i in range(self.num_sectors):
                start = max(0, i - bubble_radius_idx)
                end = min(self.num_sectors, i + bubble_radius_idx + 1)
                inflated_histogram[i] = np.min(histogram[start:end])

            # 4. 안전한 길(Valley) 찾기
            safe_indices = np.where(inflated_histogram > self.obstacle_threshold_m)[0]

            if len(safe_indices) == 0:
                # 갇혔을 때: 가장 먼 곳으로 (비상)
                target_sector = np.argmax(inflated_histogram)
            else:
                # 열린 길 중, 정면과 가장 가까운 길 선택
                gaps = np.split(safe_indices, np.where(np.diff(safe_indices) != 1)[0] + 1)
                center_sector = self.num_sectors // 2
                best_gap = min(gaps, key=lambda g: abs(g[len(g)//2] - center_sector))
                target_sector = best_gap[len(best_gap) // 2]
                
            # 5. 목표 각도 계산 및 발행
            target_angle_rad = (target_sector - self.num_sectors // 2) * self.sector_size_rad
            target_angle_deg = np.rad2deg(target_angle_rad)

            self.angle_pub.publish(target_angle_deg)

        except Exception as e:
            rospy.logwarn(f"[mission4_perception_node] e: {e}")

    def spin(self):
        rospy.loginfo("Mission 4 Perception Running...")
        rospy.spin()

if __name__ == "__main__":
    try:
        Mission4PerceptionNode().spin()
    except rospy.ROSInterruptException:
        pass
