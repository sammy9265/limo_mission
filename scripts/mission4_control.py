#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import math
import numpy as np
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32

"""
미션 4: 라바콘 주행 (Control) [Gap Follower]
- 역할: '눈'(/mission4/target_angle)이 준 각도를 Pure Pursuit으로 추종
- 구독: /mission4/target_angle (Float32, 단위: 도)
- 발행: /cmd_vel
"""

class GapFollowerNode:
    def __init__(self):
        rospy.init_node("mission4_control_node")
        rospy.loginfo("Mission 4 Control (Gap Follower) Started")

        # === [튜닝 파라미터] ===
        self.wheelbase = rospy.get_param("~wheelbase", 0.2)      # m
        self.lookahead = rospy.get_param("~lookahead", 0.8)      # m
        self.linear_speed = rospy.get_param("~linear_speed", 0.1) # m/s (튜닝용 느린 속도)

        # === ROS IO ===
        self.angle_sub = rospy.Subscriber("/mission4/target_angle", Float32, self.angle_cb, queue_size=1)
        self.cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)

        self.twist = Twist()
        self.last_msg_time = rospy.Time.now()
        
        # 안전 장치
        rospy.Timer(rospy.Duration(0.1), self.check_timeout)
        rospy.on_shutdown(self.stop_robot)

    def clamp(self, n, minn, maxn):
        return max(min(maxn, n), minn)

    def angle_cb(self, msg):
        self.last_msg_time = rospy.Time.now()
        
        target_angle_deg = msg.data
        target_th = math.radians(target_angle_deg)

        # Pure Pursuit 조향각 계산
        # delta = atan(2 * L * sin(theta) / d)
        delta = math.atan2(2.0 * self.wheelbase * math.sin(target_th), self.lookahead)

        # LIMO 제어
        self.twist.linear.x = self.linear_speed
        # 각속도 w = (v / L) * tan(delta)
        self.twist.angular.z = (self.linear_speed / self.wheelbase) * math.tan(delta)
        self.twist.angular.z = self.clamp(self.twist.angular.z, -1.5, 1.5)

        self.cmd_vel_pub.publish(self.twist)

    def check_timeout(self, event):
        if (rospy.Time.now() - self.last_msg_time).to_sec() > 1.0:
            rospy.logwarn_throttle(1.0, "M4 Timeout: No target angle. Stopping.")
            self.stop_robot()

    def stop_robot(self):
        self.twist.linear.x = 0.0
        self.twist.angular.z = 0.0
        self.cmd_vel_pub.publish(self.twist)

    def spin(self):
        rospy.spin()

if __name__ == "__main__":
    try:
        GapFollowerNode().spin()
    except rospy.ROSInterruptException:
        pass
