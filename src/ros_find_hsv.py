#!/usr/bin/env python
# -*- coding: utf-8 -*- # 한글 주석쓰려면 이거 해야함

# Standard libraries
from argparse import ArgumentParser

# ROS libraries
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage

# Other libraries
import cv2
import numpy as np


def nothing():
    pass   


def main(node, subscriber):
    """Creates a camera calibration node and keeps it running."""

    # Initialize node
    rospy.init_node(node)

    # Initialize CV Bridge
    bridge = CvBridge()

    # Create a named window to calibrate HSV values in
    cv2.namedWindow('HSV Calibrator')

    # Creating track bar
    cv2.createTrackbar('H_low_white', 'HSV Calibrator', 0, 180, nothing)
    cv2.createTrackbar('S_low_white', 'HSV Calibrator', 0, 255, nothing)
    cv2.createTrackbar('V_low_white', 'HSV Calibrator', 0, 255, nothing)

    cv2.createTrackbar('H_high_white', 'HSV Calibrator', 0, 180, nothing)
    cv2.createTrackbar('S_high_white', 'HSV Calibrator', 0, 255, nothing)
    cv2.createTrackbar('V_high_white', 'HSV Calibrator', 0, 255, nothing)

    cv2.createTrackbar('H_low_yellow', 'HSV Calibrator', 0, 180, nothing)
    cv2.createTrackbar('S_low_yellow', 'HSV Calibrator', 0, 255, nothing)
    cv2.createTrackbar('V_low_yellow', 'HSV Calibrator', 0, 255, nothing)

    cv2.createTrackbar('H_high_yellow', 'HSV Calibrator', 0, 180, nothing)
    cv2.createTrackbar('S_high_yellow', 'HSV Calibrator', 0, 255, nothing)
    cv2.createTrackbar('V_high_yellow', 'HSV Calibrator', 0, 255, nothing)


    cv2.setTrackbarPos('H_low_white', 'HSV Calibrator', 0)
    cv2.setTrackbarPos('H_high_white', 'HSV Calibrator',180)
    cv2.setTrackbarPos('S_low_white', 'HSV Calibrator', 0)
    cv2.setTrackbarPos('S_high_white', 'HSV Calibrator',255)
    cv2.setTrackbarPos('V_low_white', 'HSV Calibrator', 0)
    cv2.setTrackbarPos('V_high_white', 'HSV Calibrator',255)

    cv2.setTrackbarPos('H_low_yellow', 'HSV Calibrator', 0)
    cv2.setTrackbarPos('H_high_yellow', 'HSV Calibrator',180)
    cv2.setTrackbarPos('S_low_yellow', 'HSV Calibrator', 0)
    cv2.setTrackbarPos('S_high_yellow', 'HSV Calibrator',255)
    cv2.setTrackbarPos('V_low_yellow', 'HSV Calibrator', 0)
    cv2.setTrackbarPos('V_high_yellow', 'HSV Calibrator',255)

    # Subscribe to the specified ROS topic and process it continuously
    # rospy.Subscriber(subscriber, Image, calibrator, callback_args=(bridge))

    while not rospy.is_shutdown():
        data = rospy.wait_for_message('/camera2/usb_cam2/image_raw/compressed', CompressedImage)
        raw = bridge.compressed_imgmsg_to_cv2(data, "bgr8")
        hsv = cv2.cvtColor(raw, cv2.COLOR_BGR2HSV)

        # get info from track bar and appy to result
        white_h_low = cv2.getTrackbarPos('H_low_white', 'HSV Calibrator')
        white_s_low = cv2.getTrackbarPos('S_low_white', 'HSV Calibrator')
        white_v_low = cv2.getTrackbarPos('V_low_white', 'HSV Calibrator')
        white_h_high = cv2.getTrackbarPos('H_high_white', 'HSV Calibrator')
        white_s_high = cv2.getTrackbarPos('S_high_white', 'HSV Calibrator')
        white_v_high = cv2.getTrackbarPos('V_high_white', 'HSV Calibrator')

        yellow_h_low = cv2.getTrackbarPos('H_low_yellow', 'HSV Calibrator')
        yellow_s_low = cv2.getTrackbarPos('S_low_yellow', 'HSV Calibrator')
        yellow_v_low = cv2.getTrackbarPos('V_low_yellow', 'HSV Calibrator')
        yellow_h_high = cv2.getTrackbarPos('H_high_yellow', 'HSV Calibrator')
        yellow_s_high = cv2.getTrackbarPos('S_high_yellow', 'HSV Calibrator')
        yellow_v_high = cv2.getTrackbarPos('V_high_yellow', 'HSV Calibrator')

        # Normal masking algorithm
        white_lower = np.array([white_h_low, white_s_low, white_v_low])
        white_upper = np.array([white_h_high, white_s_high, white_v_high])

        yellow_lower = np.array([yellow_h_low, yellow_s_low, yellow_v_low])
        yellow_upper = np.array([yellow_h_high, yellow_s_high, yellow_v_high])

        white_mask = cv2.inRange(hsv, white_lower, white_upper)
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)

        white_result = cv2.bitwise_and(raw, raw, mask=white_mask)
        yellow_result = cv2.bitwise_and(raw, raw, mask=yellow_mask)

        result = cv2.bitwise_or(white_result, yellow_result)
        # return result
        # cv2.imshow('HSV Calibrator', result)
        # do stuff
        # cv2.imshow('HSV Calibrator', calibrator())
        binary = cv2.hconcat([white_result, yellow_result])
        cv2.imshow('white_yellow', binary)
        cv2.imshow('HSV Calibrator', result)
        cv2.waitKey(1)


if __name__ == "__main__":
    PARSER = ArgumentParser()
    PARSER.add_argument("--subscribe", "-s",
                        default="/cameras/lmy_cam",
                        help="ROS topic to subcribe to (str)", type=str)
    PARSER.add_argument("--node", "-n", default="CameraCalibrator",
                        help="Node name (str)", type=str)
    ARGS = PARSER.parse_args()

    main(ARGS.node, ARGS.subscribe)