#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import math
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32

"""
미션 4: 라바콘 주행 (Perception) [Gap Finder]
- 역할: 라이다 스캔 -> 장애물 버블링 -> 갭 찾기 -> 최적 갭 선택
- 발행: /mission4/target_angle (Float32, 단위: 도)
"""

class GapFinderNode:
    def __init__(self):
        rospy.init_node('mission4_perception_node')
        
        # === [튜닝 파라미터] ===
        self.scan_topic = rospy.get_param("~scan_topic", "/scan")
        self.lookahead = rospy.get_param("~lookahead", 0.8)         # m
        self.free_thresh = rospy.get_param("~free_thresh", 0.5)     # m (이 거리 이상 비어야 길로 인정)
        self.roi_front = rospy.get_param("~roi_front", 1.5)         # m (전방 인식 거리)
        self.roi_half_width = rospy.get_param("~roi_half_width", 0.6) # m (좌우 인식 폭)
        self.bias_to_heading = rospy.get_param("~bias_to_heading", 0.5) # 중앙 선호도
        self.obstacle_dist_thresh = rospy.get_param("~obstacle_dist_thresh", 0.8) # m
        self.min_gap_width = rospy.get_param("~min_gap_width", 0.35) # m (최소 통과 폭)
        
        # === ROS IO ===
        self.angle_pub = rospy.Publisher('/mission4/target_angle', Float32, queue_size=1)
        self.scan_sub = rospy.Subscriber(self.scan_topic, LaserScan, self.scan_cb, queue_size=1)
        
        rospy.loginfo("Mission 4 Perception (Gap Finder) Started")

    def scan_cb(self, msg):
        ranges = np.array(msg.ranges)
        ranges = np.nan_to_num(ranges, nan=msg.range_max, posinf=msg.range_max, neginf=msg.range_min)
        ranges = np.clip(ranges, msg.range_min, msg.range_max)
        
        N = len(ranges)
        if N == 0: return

        # 스무딩
        s = np.zeros_like(ranges)
        s[1:-1] = (ranges[:-2] + ranges[1:-1] + ranges[2:]) / 3.0
        s[0] = ranges[0]; s[-1] = ranges[-1]

        angles = msg.angle_min + np.arange(N) * msg.angle_increment

        # ROI 필터링
        x_coords = s * np.cos(angles)
        y_coords = s * np.sin(angles)
        roi_mask = (x_coords > 0.0) & (x_coords <= self.roi_front) & (np.abs(y_coords) <= self.roi_half_width)
        s[~roi_mask] = 0.0
        
        # 장애물 버블링
        valid_obstacles = s[s > 0.0]
        if len(valid_obstacles) > 0:
            min_r = np.min(valid_obstacles)
            min_idx = np.where(s == min_r)[0][0]
            safe_radius = 0.3 # m
            ang_pad = math.atan2(safe_radius, max(0.1, min_r))
            pad_idx = int(math.ceil(ang_pad / msg.angle_increment))
            start_idx = max(0, min_idx - pad_idx)
            end_idx = min(N, min_idx + pad_idx + 1)
            s[start_idx:end_idx] = 0.0

        # 갭 찾기
        is_free = (s > 0.0) & (s > self.free_thresh)
        diff = np.diff(is_free.astype(int))
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0]
        if is_free[0]: starts = np.insert(starts, 0, 0)
        if is_free[-1]: ends = np.append(ends, N-1)
        
        gaps = []
        for start, end in zip(starts, ends):
            if end <= start: continue
            mid_idx = (start + end) // 2
            d = s[mid_idx]
            th_a = angles[start]; th_b = angles[end]
            r_a = s[start]; r_b = s[end]
            p1 = np.array([r_a * math.cos(th_a), r_a * math.sin(th_a)])
            p2 = np.array([r_b * math.cos(th_b), r_b * math.sin(th_b)])
            if np.linalg.norm(p1 - p2) >= self.min_gap_width:
                ang_span = (end - start + 1) * msg.angle_increment
                ang_center = angles[mid_idx]
                heading_bias = 1.0 - self.bias_to_heading * min(1.0, abs(ang_center) / (math.pi / 2.0))
                score = ang_span * heading_bias * (0.5 + 0.5 * (d / max(0.1, self.lookahead)))
                gaps.append({'mid_idx': mid_idx, 'score': score, 'dist': d, 'angle_rad': ang_center})

        # 목표 각도 발행
        if not gaps:
            # 갈 곳 없으면 0도(직진) 혹은 정지 신호(nan) 발행
            # 여기서는 일단 0도 발행하지만, '다리'에서 거리 정보를 활용해 멈추게 할 수도 있음
            self.angle_pub.publish(0.0) 
        else:
            best_gap = max(gaps, key=lambda x: x['score'])
            # 라디안 -> 도 변환하여 발행
            target_angle_deg = math.degrees(best_gap['angle_rad'])
            self.angle_pub.publish(target_angle_deg)

    def spin(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        GapFinderNode().spin()
    except rospy.ROSInterruptException:
        pass
