#!/usr/bin/env python3
"""
Detector node - camera + odom -> YOLO + CLIP -> Zenoh.
Adds keyframe gating and CLIP embeddings on top of the A2 detector.
"""

import json
import math
import uuid
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from tf2_ros import Buffer, TransformListener
from ultralytics import YOLO
import open_clip
import torch
from PIL import Image as PILImage

import zenoh


MIN_DIST_M = 0.5
MIN_ANGLE_DEG = 15
RATE_LIMIT_S = 0.1


class DetectionNode(Node):
    def __init__(self):
        super().__init__("detection_node")

        # models
        self.yolo = YOLO("yolov8n.pt")
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        self.clip_model.eval()

        self.run_id = str(uuid.uuid4())
        self.robot_id = "tb3_sim"
        self.kf_id = 0  # keyframe counter

        # state for gating
        self.prev_x = None
        self.prev_y = None
        self.prev_yaw = None
        self.last_ts = 0.0

        # odom cache
        self.latest_odom = None
        self.tf_buf = Buffer()
        self.tf_lis = TransformListener(self.tf_buf, self)

        self.create_subscription(Odometry, "/odom", self._odom_cb, 10)
        self.create_subscription(Image, "/camera/image_raw", self._img_cb, 10)

        # zenoh
        self.zs = zenoh.open(zenoh.Config())
        self.get_logger().info(f"Detector started  run={self.run_id}")



    def _odom_cb(self, msg):
        self.latest_odom = msg

    def _img_cb(self, msg):
        if self.latest_odom is None:
            return

        # rate limit
        now = time.time()
        if now - self.last_ts < RATE_LIMIT_S:
            return
        self.last_ts = now

        odom = self.latest_odom
        pos = odom.pose.pose.position
        q = odom.pose.pose.orientation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y ** 2 + q.z ** 2),
        )

        # keyframe gate
        if self.prev_x is not None:
            dist = math.hypot(pos.x - self.prev_x, pos.y - self.prev_y)
            dang = abs(yaw - self.prev_yaw)
            if dang > math.pi:
                dang = 2 * math.pi - dang
            if dist < MIN_DIST_M and dang < math.radians(MIN_ANGLE_DEG):
                return  # not a keyframe, skip

        self.kf_id += 1
        self.prev_x, self.prev_y, self.prev_yaw = pos.x, pos.y, yaw

        # decode image
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, 3
        )

        # YOLO
        results = self.yolo(img, verbose=False)
        dets = []
        crops = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = [int(c) for c in box.xyxy[0]]
                crop = img[max(0, y1):y2, max(0, x1):x2]
                if crop.size == 0:
                    continue
                crops.append(crop)
                dets.append({
                    "det_id": str(uuid.uuid4()),
                    "class_id": int(box.cls[0]),
                    "class_name": self.yolo.names[int(box.cls[0])],
                    "confidence": round(float(box.conf[0]), 4),
                    "bbox_xyxy": [round(float(c), 2) for c in box.xyxy[0]],
                })

        if not dets:
            return

        # CLIP embeddings
        pil_crops = [PILImage.fromarray(c) for c in crops]
        batch = torch.stack([self.preprocess(c) for c in pil_crops])
        with torch.no_grad():
            embs = self.clip_model.encode_image(batch)
            embs = embs / embs.norm(dim=-1, keepdim=True)
            embs = embs.cpu().numpy()

        for i, d in enumerate(dets):
            d["embedding"] = embs[i].tolist()
            d["embedding_dim"] = 512
            d["embedding_model"] = "ViT-B/32"

        # TF (best effort)
        tf_ok = False
        t_bc = [0.0] * 16
        try:
            t = self.tf_buf.lookup_transform(
                "base_footprint", "camera_link", rclpy.time.Time()
            )
            tf_ok = True
            tr = t.transform.translation
            t_bc = [1, 0, 0, tr.x, 0, 1, 0, tr.y, 0, 0, 1, tr.z, 0, 0, 0, 1]
        except Exception:
            pass

        event = {
            "schema": "maze.detection.v2",
            "event_id": str(uuid.uuid4()),
            "run_id": self.run_id,
            "robot_id": self.robot_id,
            "keyframe_id": self.kf_id,
            "sequence": self.kf_id,
            "image": {
                "stamp": {
                    "sec": msg.header.stamp.sec,
                    "nanosec": msg.header.stamp.nanosec,
                },
                "frame_id": msg.header.frame_id,
                "width": msg.width,
                "height": msg.height,
                "encoding": msg.encoding,
            },
            "odometry": {
                "map_x": round(pos.x, 4),
                "map_y": round(pos.y, 4),
                "map_yaw": round(yaw, 4),
            },
            "tf": {
                "tf_ok": tf_ok,
                "t_base_camera": t_bc,
            },
            "detections": dets,
        }

        key = f"maze/{self.robot_id}/{self.run_id}/detections/v2/{event['event_id']}"
        self.zs.put(key, json.dumps(event).encode())
        self.get_logger().info(
            f"KF {self.kf_id}: {len(dets)} dets at ({pos.x:.1f},{pos.y:.1f})"
        )


def main():
    rclpy.init()
    node = DetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.zs.close()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
