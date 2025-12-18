#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import numpy as np
import cv2

class EdgeLaneNoBridge:
def __init__(self):
rospy.init_node("edge_lane_nobridge_node")

# Subscriber & Publisher
rospy.Subscriber("/usb_cam/image_raw", Image, self.image_callback, queue_size=1)
self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=3)

self.cmd = Twist()
self.current_lin = 0.0
self.current_ang = 0.0

self.encoding = None

# ===== 튜닝 파라미터 (예전에 잘 움직이던 쪽에 가깝게) =====
self.forward_speed = 0.12 # 기본 전진 속도
self.search_spin_speed = 0.25 # 라인 못 찾을 때 회전 속도

self.canny_low = 50
self.canny_high = 150
self.edge_thresh = 10 # 열당 엣지 개수 최대값이 이거보다 커야 "라인 있다"로 판단
self.k_angle = 0.010 # 조향 게인

rospy.loginfo("✅ EdgeLaneNoBridge node started (simple EDGE mode)")

# ----------------------------- #
# Image msg -> numpy (cv2용)
# ----------------------------- #
def msg_to_cv2(self, msg: Image):
if self.encoding is None:
self.encoding = msg.encoding
rospy.loginfo("📷 image encoding: %s", self.encoding)

h = msg.height
w = msg.width

# 3채널 영상 (rgb8/bgr8)
if self.encoding in ("rgb8", "bgr8"):
arr = np.frombuffer(msg.data, dtype=np.uint8)
try:
img = arr.reshape(h, msg.step // 3, 3)
img = img[:, :w, :]
except Exception as e:
rospy.logwarn("reshape error: %s", e)
return None

if self.encoding == "rgb8":
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
return img

# 1채널 영상 (mono8)
if self.encoding == "mono8":
arr = np.frombuffer(msg.data, dtype=np.uint8)
try:
img = arr.reshape(h, msg.step)
img = img[:, :w]
except Exception as e:
rospy.logwarn("reshape mono8 error: %s", e)
return None
return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

rospy.logwarn_throttle(2.0, "Unsupported encoding: %s", self.encoding)
return None

# ----------------------------- #
# 이미지 콜백: "검은 트랙" 중앙 추종 버전
# ----------------------------- #
def image_callback(self, msg: Image):
img = self.msg_to_cv2(msg)
if img is None:
# 이미지 못 읽으면 회전만
self.current_lin = 0.0
self.current_ang = self.search_spin_speed
return

h, w, _ = img.shape
center = w / 2.0

# 1) 바닥 쪽 ROI (하단 50% 사용해서 트랙 폭 넓게 보기)
roi_y_start = int(h * 0.5) # 필요하면 0.4~0.6 사이에서 튜닝
roi = img[roi_y_start:, :]

# 2) 그레이 + 블러
gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)

# 3) 검은 트랙 강조: THRESH_BINARY_INV + OTSU
# → 어두운 부분(트랙)이 255, 나머지는 0
_, binary = cv2.threshold(
gray, 0, 255,
cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

# 4) 노이즈 제거 (3x3 작은 커널만)
kernel = np.ones((3, 3), np.uint8)
binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

# 5) 열별 "검은 픽셀(=255)" 개수
mask = (binary > 0)
col_sum = np.sum(mask, axis=0) # shape: (w,)
max_val = int(np.max(col_sum)) if col_sum.size > 0 else 0

# 너무 어둡게 안 잡히면 트랙 못 찾았다고 보고 회전
dark_min_pixels = 5 # 너무 적으면 1~3, 너무 많으면 10 이상으로 튜닝
if max_val < dark_min_pixels:
self.current_lin = 0.0
self.current_ang = self.search_spin_speed
rospy.loginfo_throttle(
0.8,
f"[BLACK] no dark enough column (max={max_val}) → spin"
)
return

# 6) max의 일정 비율 이상인 열들만 "트랙 후보"로 사용
dark_col_ratio = 0.3 # 0.2~0.5 사이에서 튜닝
threshold_val = max(dark_min_pixels, int(max_val * dark_col_ratio))
candidates = np.where(col_sum >= threshold_val)[0]

if candidates.size == 0:
self.current_lin = 0.0
self.current_ang = self.search_spin_speed
rospy.loginfo_throttle(
0.8,
f"[BLACK] no candidate columns (max={max_val}) → spin"
)
return

# 7) 후보 열들의 무게중심 = 검은 트랙 중앙 x
x = np.arange(len(col_sum))
track_center_x = float(np.sum(x[candidates] * col_sum[candidates]) /
np.sum(col_sum[candidates]))

offset = track_center_x - center # +: 오른쪽에 트랙, -: 왼쪽에 트랙
offset_norm = offset / (w / 2.0)

# 예전에 쓰던 조향 부호 유지: 왼쪽이면 +, 오른쪽이면 - (필요하면 부호만 바꿔서 튜닝)
ang = -self.k_angle * offset
ang = max(min(ang, 0.8), -0.8)

self.current_lin = self.forward_speed
self.current_ang = ang

rospy.loginfo_throttle(
0.3,
f"[BLACK] center={track_center_x:.1f} off={offset:.1f} "
f"norm={offset_norm:.2f} w={ang:.3f} max={max_val} cand={candidates.size}"
)

# ----------------------------- #
# /cmd_vel 계속 발행
# ----------------------------- #
def spin(self):
rate = rospy.Rate(20)
while not rospy.is_shutdown():
self.cmd.linear.x = self.current_lin
self.cmd.angular.z = self.current_ang
self.cmd_pub.publish(self.cmd)
rate.sleep()

if __name__ == "__main__":
node = EdgeLaneNoBridge()
try:
node.spin()
except rospy.ROSInterruptException:
pass

