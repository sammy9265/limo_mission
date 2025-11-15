#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from geometry_msgs.msg import PointStamped, Twist

"""
미션 1: 라인 트레이싱 (Control) 노드
- 역할: 'mission1_perception'이 발행한 중심점을 받아 LIMO를 제어
- 구독: /perception/center_point_px (차선 중심점)
- 발행: /cmd_vel (LIMO 바퀴 제어)
"""

class LineControlNode:
    def __init__(self):
        rospy.init_node("line_control_node")
        rospy.loginfo("Line Control Node (Mission 1) started")

        # === ROS IO ===
        self.center_sub = rospy.Subscriber(
            "/perception/center_point_px", 
            PointStamped, 
            self.cb_center, 
            queue_size=1
        )
        self.cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)

        # === [튜닝 대상 4 - rosrun 파라미터] (제어 목표) ===
        # LIMO가 따라가야 할 BEV 이미지의 중앙 x좌표 (예: 640x480 이미지의 중앙 320)
        self.target_x_px = rospy.get_param("~target_x_px", 320.0) 
        
        # === [튜닝 대상 5 - rosrun 파라미터] (속도) ===
        # LIMO의 기본 주행 속도 (m/s), (처음엔 0.1 ~ 0.2 로 낮게 시작)
        self.linear_speed = rospy.get_param("~linear_speed", 0.1)

        # === [튜닝 대상 6 - rosrun 파라미터] (PD 제어 게인) ===
        # P: 오차에 비례해 핸들을 꺾는 강도. (클수록 민감)
        # D: 급격한 핸들 꺾임을 방지(진동 억제).
        # (튜닝 순서: Kp부터 맞추고, 진동하면 Kd 값을 올림)
        self.kp = rospy.get_param("~kp", 0.004) # <-- P 게인
        self.kd = rospy.get_param("~kd", 0.001) # <-- D 게인

        # === 제어 변수 ===
        self.last_error = 0.0
        self.last_msg_time = rospy.Time.now()
        self.twist_msg = Twist()

        # [안전 장치]
        rospy.Timer(rospy.Duration(0.1), self.check_timeout)
        rospy.on_shutdown(self.stop_robot) # Ctrl+C 누르면 정지

    def cb_center(self, msg):
        self.last_msg_time = rospy.Time.now()

        # 1. 오차(Error) 계산 = 목표지점(중앙) - 현재 차선 중심
        current_x = msg.point.x
        error = self.target_x_px - current_x
        
        # 2. PD 제어 계산
        error_diff = error - self.last_error  # 오차의 변화량 (D항)
        
        # 제어량(조향각) 계산
        angular_z = -1.0 * (self.kp * error + self.kd * error_diff)

        # 3. Twist 메시지 발행
        self.twist_msg.linear.x = self.linear_speed
        self.twist_msg.angular.z = angular_z
        self.cmd_vel_pub.publish(self.twist_msg)

        # 4. 다음 계산을 위해 현재 오차 저장
        self.last_error = error

    def check_timeout(self, event):
        # 0.5초 동안 차선 중심점 메시지가 안 들어오면 정지
        if (rospy.Time.now() - self.last_msg_time).to_sec() > 0.5:
            rospy.logwarn("Timeout: No center point received. Stopping robot.")
            self.stop_robot()

    def stop_robot(self):
        self.twist_msg.linear.x = 0.0
        self.twist_msg.angular.z = 0.0
        self.cmd_vel_pub.publish(self.twist_msg)
        rospy.loginfo("Robot stopped.")

    def spin(self):
        rospy.loginfo("Line Control Node (Mission 1) running...")
        rospy.spin()

if __name__ == "__main__":
    try:
        LineControlNode().spin()
    except rospy.ROSInterruptException:
        pass