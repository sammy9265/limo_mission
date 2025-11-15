#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
import threading # 스레드 충돌 방지를 위해 추가
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import Float32
from geometry_msgs.msg import PointStamped

"""
미션 1: 차선 인식 (Perception) 노드 (네트워크 지연에 강한 버전)
- 역할: LIMO의 카메라를 보고 차선 중심점을 계산
- 구독: /camera/color/image_raw (LIMO 카메라)
- 발행: /perception/center_point_px (미션 1 제어 노드가 사용할 중심점)
-
- [수정 사항]
- cb_image(콜백)와 spin(메인 루프)을 분리.
- cb_image: 이미지 처리만 하고 변수에 저장 (느린 3.5Hz로 동작)
- spin: 30Hz로 빠르게 돌면서 저장된 변수로 OpenCV 창만 갱신 (응답 없음 방지)
"""

class LanePerceptionNode:
    def __init__(self):
        rospy.init_node("mission1_perception_node") 
        rospy.loginfo("Lane Perception Node (Mission 1) [Robust Ver.] started")

        # === [튜닝 UI] ===
        self.show_window = rospy.get_param("~show_window", True) 
        self.win_src = "src_with_roi"
        self.win_bev = "bev_binary_and_windows"
        if self.show_window:
            try: cv2.startWindowThread()
            except Exception: pass
            cv2.namedWindow(self.win_src, cv2.WINDOW_NORMAL); cv2.resizeWindow(self.win_src, 960, 540)
            cv2.namedWindow(self.win_bev, cv2.WINDOW_NORMAL); cv2.resizeWindow(self.win_bev, 960, 540)

        # === 스레드 안전을 위한 변수 ===
        self.lock = threading.Lock()
        self.latest_src_vis = None  # src_with_roi 창에 표시할 이미지
        self.latest_bev_binary = None # bev_binary_and_windows 창에 표시할 이미지
        self.latest_debug_img = None  # bev_binary_and_windows 창에 표시할 이미지 (디버그용)
        self.h, self.w = 480, 640  # 기본 이미지 크기 (첫 프레임에서 갱신됨)

        # === ROS IO ===
        self.bridge = CvBridge()
        self.sub = rospy.Subscriber("/camera/color/image_raw", Image, self.cb_image,
                                    queue_size=1, buff_size=2**24) # 큐 사이즈 1로 변경
        
        self.pub_center_point = rospy.Publisher("/perception/center_point_px", PointStamped, queue_size=1)
        self.pub_k_left   = rospy.Publisher("/perception/curvature_left",   Float32, queue_size=1)
        self.pub_k_right  = rospy.Publisher("/perception/curvature_right",  Float32, queue_size=1)
        self.pub_k_center = rospy.Publisher("/perception/curvature_center", Float32, queue_size=1)

        # === [튜닝 대상 1 - rosrun 파라미터] (차선 폭) ===
        self.lane_width_px = rospy.get_param("~lane_width_px", 340.0)

        # === [튜닝 대상 2 - rosrun 파라미터] (ROI 영역) ===
        self.roi_top_y_ratio     = rospy.get_param("~roi_top_y_ratio", 0.45) # 튜닝 시작 값
        self.roi_left_top_ratio  = rospy.get_param("~roi_left_top_ratio", 0.1)
        self.roi_right_top_ratio = rospy.get_param("~roi_right_top_ratio", 0.9)
        self.roi_left_bot_ratio  = rospy.get_param("~roi_left_bot_ratio", 0.0)
        self.roi_right_bot_ratio = rospy.get_param("~roi_right_bot_ratio", 1.0)

        # === [튜닝 대상 3 - 이 파일(VSCode)에서 직접 수정] (색상) ===
        self.yellow_lower = np.array([10,  80,  60], dtype=np.uint8)
        self.yellow_upper = np.array([45, 255, 255], dtype=np.uint8)
        self.white_lower  = np.array([ 0,   0, 150], dtype=np.uint8) # 튜닝 시작 값 (160->150)
        self.white_upper  = np.array([179,  80, 255], dtype=np.uint8)

        # === 슬라이딩 윈도우 파라미터 ===
        self.num_windows = rospy.get_param("~num_windows", 12)
        self.window_margin = rospy.get_param("~window_margin", 80)
        self.minpix_recenter = rospy.get_param("~minpix_recenter", 50)
        self.min_lane_sep = rospy.get_param("~min_lane_sep", 60)
        self.center_ema_alpha = rospy.get_param("~center_ema_alpha", 0.8)

    # --- (헬퍼 함수들은 수정할 필요 없음) ---
    def make_roi_polygon(self, h, w):
        y_top = int(h * self.roi_top_y_ratio); y_bot = h - 1
        x_lt  = int(w * self.roi_left_top_ratio); x_rt  = int(w * self.roi_right_top_ratio)
        x_lb  = int(w * self.roi_left_bot_ratio); x_rb  = int(w * self.roi_right_bot_ratio)
        return np.array([[x_lb, y_bot], [x_lt, y_top], [x_rt, y_top], [x_rb, y_bot]], np.int32)

    def warp_to_bev(self, bgr, roi_poly):
        h, w = bgr.shape[:2]; BL, TL, TR, BR = roi_poly.astype(np.float32)
        for p in (BL, TL, TR, BR): p[1] = np.clip(p[1], 0, h - 1)
        src = np.float32([BL, TL, TR, BR])
        dst = np.float32([[0, h-1], [0, 0], [w-1, 0], [w-1, h-1]])
        M = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(bgr, M, (w, h), flags=cv2.INTER_LINEAR)

    def binarize_lanes(self, bgr):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask_y = cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)
        mask_w = cv2.inRange(hsv, self.white_lower,  self.white_upper)
        kernel = np.ones((3, 3), np.uint8)
        mask_y = cv2.morphologyEx(mask_y, cv2.MORPH_OPEN, kernel, iterations=1)
        mask_w = cv2.morphologyEx(mask_w, cv2.MORPH_OPEN, kernel, iterations=1)
        return cv2.bitwise_or(mask_y, mask_w)

    def run_sliding_window_collect_centers(self, binary_mask):
        h, w = binary_mask.shape[:2]; nonzero = binary_mask.nonzero()
        nz_y = np.array(nonzero[0]); nz_x = np.array(nonzero[1])
        histogram = np.sum(binary_mask[h//2:, :], axis=0); midpoint = w // 2
        left_base = np.argmax(histogram[:midpoint]) if histogram[:midpoint].any() else None
        right_base = (np.argmax(histogram[midpoint:]) + midpoint) if histogram[midpoint:].any() else None
        debug_img = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
        window_height = int(h / self.num_windows); left_current = left_base; right_current = right_base
        left_indices, right_indices, left_window_centers, right_window_centers = [], [], [], []
        for win in range(self.num_windows):
            y_low, y_high = h - (win + 1) * window_height, h - win * window_height
            if left_current is not None: cv2.rectangle(debug_img, (left_current - self.window_margin, y_low), (left_current + self.window_margin, y_high), (255, 0, 0), 2)
            if right_current is not None: cv2.rectangle(debug_img, (right_current - self.window_margin, y_low), (right_current + self.window_margin, y_high), (255, 0, 0), 2)
            good_left, good_right = [], []
            if left_current is not None: good_left = ((nz_y >= y_low) & (nz_y < y_high) & (nz_x >= left_current - self.window_margin) & (nz_x <  left_current + self.window_margin)).nonzero()[0].tolist()
            if right_current is not None: good_right = ((nz_y >= y_low) & (nz_y < y_high) & (nz_x >= right_current - self.window_margin) & (nz_x <  right_current + self.window_margin)).nonzero()[0].tolist()
            if left_current is not None and right_current is not None and abs(left_current - right_current) < self.min_lane_sep:
                if len(good_left) < len(good_right): good_left = []
                else: good_right = []
            left_indices.extend(good_left); right_indices.extend(good_right); y_center = (y_low + y_high) // 2
            if len(good_left) > 0:
                x_mean_left = float(np.mean(nz_x[good_left])); left_window_centers.append((int(y_center), float(x_mean_left)))
                cv2.circle(debug_img, (int(x_mean_left), int(y_center)), 4, (0, 0, 255), -1)
            if len(good_right) > 0:
                x_mean_right = float(np.mean(nz_x[good_right])); right_window_centers.append((int(y_center), float(x_mean_right)))
                cv2.circle(debug_img, (int(x_mean_right), int(y_center)), 4, (0, 255, 255), -1)
            if len(good_left) > self.minpix_recenter and left_current is not None: left_current = int(self.center_ema_alpha * left_current + (1 - self.center_ema_alpha) * float(np.mean(nz_x[good_left])))
            if len(good_right) > self.minpix_recenter and right_current is not None: right_current = int(self.center_ema_alpha * right_current + (1 - self.center_ema_alpha) * float(np.mean(nz_x[good_right])))
        if len(left_indices) > 0: debug_img[np.clip(nz_y[left_indices], 0, h-1), np.clip(nz_x[left_indices], 0, w-1)] = (0, 0, 255)
        if len(right_indices) > 0: debug_img[np.clip(nz_y[right_indices], 0, h-1), np.clip(nz_x[right_indices], 0, w-1)] = (0, 255, 0)
        return debug_img, left_window_centers, right_window_centers

    def compute_curvature_from_centers(self, centers, image_height):
        if len(centers) < 5: return None, None
        ys = np.array([p[0] for p in centers], dtype=np.float64); xs = np.array([p[1] for p in centers], dtype=np.float64)
        fit = np.polyfit(ys, xs, 2); a, b, c = fit; y_eval = float(image_height - 1)
        dxdy = 2*a*y_eval + b; d2xdy2 = 2*a
        return fit, abs(d2xdy2) / ((1.0 + dxdy*dxdy) ** 1.5)

    def compute_center_point(self, left_window_centers, right_window_centers, image_height):
        def side_mean(centers):
            if not centers: return None
            arr = np.array(centers, dtype=np.float64)
            return (int(round(float(np.mean(arr[:, 0])))), float(np.mean(arr[:, 1])))
        left_rep, right_rep = side_mean(left_window_centers), side_mean(right_window_centers)
        half_w = 0.5 * float(self.lane_width_px)
        if left_rep and right_rep: return (int(round(0.5 * (left_rep[0] + right_rep[0]))), 0.5 * (left_rep[1] + right_rep[1]))
        if left_rep: return (left_rep[0], left_rep[1] + half_w)
        if right_rep: return (right_rep[0], right_rep[1] - half_w)
        return None

    def draw_polynomial(self, canvas, fit, color=(255, 255, 0), step=10):
        h = canvas.shape[0]; ploty = np.arange(0, h, step, dtype=np.int32)
        fitx = (fit[0]*ploty**2 + fit[1]*ploty + fit[2]).astype(np.int32)
        for y, x in zip(ploty, fitx): cv2.circle(canvas, (int(x), int(y)), 2, color, -1)

    def cb_image(self, msg):
        """
        카메라 이미지가 3.5Hz로 느리게 들어와도, 이 함수는 처리만 하고 
        창 갱신(imshow)은 spin() 함수에 맡깁니다.
        """
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            if bgr is None or bgr.ndim == 2 or (bgr.ndim == 3 and bgr.shape[2] == 1): return
            
            self.h, self.w = bgr.shape[:2] # h, w 갱신
            roi_poly = self.make_roi_polygon(self.h, self.w)
            
            # --- 시각화용 이미지 생성 1 ---
            src_vis = bgr.copy(); overlay = bgr.copy(); cv2.fillPoly(overlay, [roi_poly], (0, 255, 0))
            src_vis = cv2.addWeighted(overlay, 0.25, bgr, 0.75, 0); cv2.polylines(src_vis, [roi_poly], True, (0, 0, 0), 2)
            
            # --- 핵심 처리 ---
            bev_bgr = self.warp_to_bev(bgr, roi_poly); bev_binary = self.binarize_lanes(bev_bgr)
            debug_img, left_centers, right_centers = self.run_sliding_window_collect_centers(bev_binary)
            left_fit, left_k = self.compute_curvature_from_centers(left_centers, self.h)
            right_fit, right_k = self.compute_curvature_from_centers(right_centers, self.h)
            
            center_fit, center_k = None, None
            if left_fit is not None and right_fit is not None:
                center_fit = 0.5*(left_fit + right_fit); _, center_k = self.compute_curvature_from_centers([ (y, center_fit[0]*y**2 + center_fit[1]*y + center_fit[2]) for y in range(0, self.h, 10)], self.h)
            
            def curv_msg(v): return Float32(data=float(v)) if (v is not None and np.isfinite(float(v))) else Float32(data=float('nan'))
            self.pub_k_left.publish(curv_msg(left_k)); self.pub_k_right.publish(curv_msg(right_k)); self.pub_k_center.publish(curv_msg(center_k))
            
            center_point = self.compute_center_point(left_centers, right_centers, self.h)
            if center_point is not None:
                cy, cx = center_point; pt_msg = PointStamped(); pt_msg.header.stamp = msg.header.stamp
                pt_msg.header.frame_id = "bev"; pt_msg.point.x = float(cx); pt_msg.point.y = float(cy)
                self.pub_center_point.publish(pt_msg)
                cv2.circle(debug_img, (int(cx), int(cy)), 6, (255, 0, 255), -1) # 디버그 이미지에 원 그리기

            # --- 시각화용 이미지 생성 2 ---
            if left_fit is not None: self.draw_polynomial(debug_img, left_fit, (0, 0, 255))
            if right_fit is not None: self.draw_polynomial(debug_img, right_fit, (0, 255, 0))
            def put(txt, y): cv2.putText(debug_img, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
            put(f"L:{len(left_centers)} R:{len(right_centers)}", 24)
            
            # --- [수정] 스레드 잠금 후, spin()이 사용할 최신 이미지 저장 ---
            with self.lock:
                self.latest_src_vis = src_vis
                self.latest_bev_binary = bev_binary # bev 창에는 순수 이진화 이미지만 표시
                self.latest_debug_img = debug_img   # src 창에는 디버그 이미지 표시 (창 2개로 분리)

        except Exception as e: 
            rospy.logwarn(f"[lane_perception_node] cb_image e: {e}")

    def spin(self):
        """
        rospy.spin() 대신, 30Hz 루프를 돌면서 OpenCV 창을 강제로 갱신합니다.
        (응답 없음 에러 방지)
        """
        rospy.loginfo("Lane Perception Node (Mission 1) running...")
        rate = rospy.Rate(30) # 30Hz

        while not rospy.is_shutdown():
            if self.show_window:
                # 스레드 잠금 후, 최신 이미지를 가져와서 창에 표시
                with self.lock:
                    if self.latest_src_vis is not None and self.latest_debug_img is not None:
                        # (변경) 두 창의 역할을 분리
                        # win_src: 원본 + ROI + 디버그 (슬라이딩 윈도우, 중심점)
                        # win_bev: BEV 이진화 (색상 튜닝용)
                        canvas_src = np.hstack([
                            cv2.resize(self.latest_src_vis, (self.w, self.h)),
                            cv2.resize(self.latest_debug_img, (self.w, self.h))
                        ])
                        cv2.imshow(self.win_src, canvas_src)
                    
                    if self.latest_bev_binary is not None:
                        cv2.imshow(self.win_bev, self.latest_bev_binary)
                
                # imshow를 갱신하기 위해 waitKey(1)은 루프 안에 있어야 함
                cv2.waitKey(1)
            
            rate.sleep()
        
        # 종료 시 창 닫기
        if self.show_window:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        LanePerceptionNode().spin()
    except rospy.ROSInterruptException:
        pass