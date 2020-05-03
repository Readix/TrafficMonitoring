import cv2
import numpy as np
import json
import os
import re
import yolo.darknet.darknet as darknet 
import haversine
import imutils as imu
from tracker.sort import *

#--------------------------------------------
'''
res_path - path to:
    - calibration.npy
    - perspective_matrix.npy
    - mask.png
    - sides.png
areas_path - path to json file with areas points
config_path, meta_path, weight_path - paths for yolo's data
url - path to video file or url of video stream
'''
res_path = 'resources/'
areas_path = 'areas.json'
config_path = 'yolo/configs/yolo.cfg'
meta_path = 'yolo/configs/yolo.data'
weight_path = 'yolo/configs/yolo.weights'
url = 'video/sample.mp4'
#--------------------------------------------

with open(areas_path) as fh:
    config = json.load(fh)
net_main = darknet.load_net_custom(config_path.encode('ascii'), weight_path.encode('ascii'), 0, 1)
meta_main = darknet.load_meta(meta_path.encode('ascii'))

colors = {
    '': (0, 0, 0),
    'car': (0, 255, 255),
    'mini_bus': (255, 0, 0),
    'bus': (255, 0, 255),
    'truck': (0, 0, 255),
    'tram': (203, 192, 255),
    'trolleybus': (0, 255, 0),
}

width = darknet.network_width(net_main)
height = darknet.network_height(net_main)
darknet_image = darknet.make_image(width, height, 3)
mask = cv2.imread(res_path + 'mask.jpg')

areas = config['areas']
persp_mtx = np.load(res_path + 'perspective_matrix.npy')
sides_png = cv2.resize(cv2.imread(res_path + 'sides.png'), (110, 55))

counters = dict()
for side in ['from', 'to']:
    counters[side] = dict()
    for area in areas:
        counters[side][area['description']] = set()

camera_data = {
    'cars_tracks': {},
    'sort_tracker': Sort(),
}


def convert_back(x, y, w, h):
    x1 = int(round(x - (w / 2)))
    y1 = int(round(y - (h / 2)))
    x2 = int(round(x + (w / 2)))
    y2 = int(round(y + (h / 2)))
    return x1, y1, x2, y2


def get_detections(image):
    frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (darknet.network_width(net_main), darknet.network_height(net_main)), interpolation=cv2.INTER_LINEAR)
    darknet_frame = cv2.addWeighted(frame_resized, 1, mask, 1, 0, 3)

    darknet.copy_image_from_bytes(darknet_image, darknet_frame.tobytes())
    detections = darknet.detect_image(net_main, meta_main, darknet_image, thresh=0.15)

    converted_detections = []
    for detection in detections:
        x, y, w, h, score, obj_type = detection[2][0], detection[2][1], detection[2][2], detection[2][3], detection[1], detection[0]
        x1, y1, x2, y2 = convert_back(float(x), float(y), float(w), float(h))
        converted_detections.append([x1 / width, y1 / height, x2 / width, y2 / height, score, obj_type.decode()])

    return converted_detections


def analysis_and_display(detections, image):
    cars_tracks = camera_data['cars_tracks']
    sort_tracker = camera_data['sort_tracker']

    trackers = sort_tracker.update(np.array(
        list(map(lambda d: [*d[:-1]], detections))))
    
    filtered_detections = []
    for tracker in trackers:
        x1, y1, x2, y2 = tracker[0], tracker[1], tracker[2], tracker[3]
        cx1 = x1 + (x2 - x1) / 2
        cy1 = y1 + (y2 - y1) / 2
        min_detection = None
        min_dist = -1.0
        for detection in detections:
            cx2 = detection[0] + (detection[2] - detection[0]) / 2
            cy2 = detection[1] + (detection[3] - detection[1]) / 2
            dist = ((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2) ** 0.5
            if min_dist == -1 or dist < min_dist:
                min_dist = dist
                min_detection = detection
        if min_detection is not None:
            filtered_detections.append({
                'id': int(tracker[4]),
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2,
                'score': min_detection[4],
                'type': min_detection[5]
            })

    for detection in filtered_detections:
        x1, y1, x2, y2 = detection['x1'], detection['y1'], detection['x2'], detection['y2']
        cx = x1 + (x2 - x1) / 2
        cy = y1 + (y2 - y1) / 2
        original = np.array([((0.7, 0.4), (cx, cy))], dtype=np.float32)
        converted = cv2.perspectiveTransform(original, persp_mtx)
        detection['lat'] = float(converted[0][1][0])
        detection['lng'] = float(converted[0][1][1])
        point = Point(cx, cy)
        for area in areas:
            points = list(map(lambda p: (p['x'], p['y']), area['in']))
            polygon = Polygon(points)
            if polygon.contains(point):
                detection['zone'] = area['description'] + ' in'
                detection['zone_side'] = area['description']
                detection['zone_direction'] = 'in'
            points = list(map(lambda p: (p['x'], p['y']), area['out']))
            polygon = Polygon(points)
            if polygon.contains(point):
                detection['zone'] = area['description'] + ' out'
                detection['zone_side'] = area['description']
                detection['zone_direction'] = 'out'
        car_id = detection['id']
        if car_id in cars_tracks:
            car_tracks = cars_tracks[car_id]
        else:
            cars_tracks[car_id] = car_tracks = []
        car_track = detection.copy()
        car_track['millis'] = millis
        car_tracks.append(car_track)
        if len(car_tracks) > 7:
            lat1, lng1, millis1 = car_tracks[-8]['lat'], car_tracks[-8]['lng'], car_tracks[-8]['millis']
            lat2, lng2, millis2 = car_tracks[-1]['lat'], car_tracks[-1]['lng'], car_tracks[-1]['millis']
            dist = haversine.haversine((lat1, lng1), (lat2, lng2))
            dt = (millis2 - millis1) / 3600000
            car_tracks[-1]['speed'] = dist / dt
            detection['speed'] = dist / dt

    now = time.time()
    new_cars = []
    for car_id in list(cars_tracks):
        car_tracks = cars_tracks[car_id]
        last_car_track = car_tracks[-1]
        seconds = int(last_car_track["millis"] / 1000)
        if now - seconds > 3:
            del cars_tracks[car_id]

            zone_from = None
            zone_to = None
            for car_track in car_tracks:
                if "zone_direction" in car_track and "zone_side" in car_track:
                    if car_track["zone_direction"] == "in":
                        zone_from = car_track["zone_side"]
                    if car_track["zone_direction"] == "out":
                        zone_to = car_track["zone_side"]

                new_cars.append({
                    'id': car_id,
                    "seconds": seconds,
                    "from": zone_from,
                    "to": zone_to,
                    "type": last_car_track["type"]
                })

    cv2.rectangle(image, (0, 0), (220, 150), (0,0,0), -1)
    for new_car in new_cars:
        if new_car['to'] is not None:
            counters['to'][new_car['to']].add(new_car['id'])
        if new_car['from'] is not None:
            counters['from'][new_car['from']].add(new_car['id'])

    cv2.putText(image, 'From', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255, 255, 255], 1)
    cv2.putText(image, 'To', (130, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255, 255, 255], 1)
    for i, side in enumerate(counters['to']):
        text = side + ': ' + str(len(counters['to'][side]))
        cv2.putText(image, text, (130, 50 + 30*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255, 255, 255], 1)
    for i, side in enumerate(counters['from']):
        text = side + ': ' + str(len(counters['from'][side]))
        cv2.putText(image, text, (20, 50 + 30*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255, 255, 255], 1)

    for detection in filtered_detections:
        h, w, _ = image.shape
        cv2.rectangle(image, (int(detection['x1'] * w), int(detection['y1'] * h)), (int(detection['x2'] * w), int(detection['y2'] * h)), colors[detection['type']], 1)
        if 'speed' in detection:
            cv2.putText(image, str(round(detection['speed'], 1)), (int(detection['x1'] * w), int(detection['y1'] * h - 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255, 255, 255], 1)

    image[:sides_png.shape[0],-sides_png.shape[1]:] = sides_png
    for area in areas:
        h, w, _ = image.shape
        contur_out = np.array(list(map(lambda p: (int(p['x'] * w), int(p['y'] * h)), area['out'])), np.int32)
        contur_in = np.array(list(map(lambda p: (int(p['x'] * w), int(p['y'] * h)), area['in'])), np.int32)
        cv2.polylines(image,[contur_out.reshape((-1,1,2))], True, (0, 0, 255))
        cv2.polylines(image,[contur_in.reshape((-1,1,2))], True, (255, 0, 0))
    cv2.imshow('frame', image)

    ch = 0xFF & cv2.waitKey(1)
    if ch == 27:
        exit(0)


if __name__ == '__main__':
    frame_skip_count = 25 / 8
    frame_ind = 0
    calibration_data = np.load(res_path + 'calibration.npy', allow_pickle=True)
    mtx = calibration_data[0]
    dist = calibration_data[1]

    while True:
        try:
            ret, frame_read = cap.read()
            if frame_read is None or not ret:
                cap = cv2.VideoCapture(url)
                continue
        except:
            cap = cv2.VideoCapture(url)
            continue

        frame_ind += 1
        if frame_ind < frame_skip_count:
            continue
        frame_ind = 0

        millis = round(time.time() * 1000)

        resized_image = cv2.resize(frame_read, (1920, 1080))
        h, w = resized_image.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))
        frame_read = cv2.undistort(resized_image, mtx, dist, None, newcameramtx)

        resized_image = imu.resize(frame_read, width=960)
        detections = get_detections(resized_image)
        analysis_and_display(detections, resized_image)
