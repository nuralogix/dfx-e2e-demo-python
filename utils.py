import asyncio
import json
import os

import numpy as np
import cv2


async def findVideoRotation(fileName):
    """Find video rotation using ffprobe."""
    angle = 0
    ffprobe_cmd = "ffprobe -v quiet -select_streams v:0 -show_entries " \
                  "stream_tags=rotate -of default=nw=1:nk=1".split(' ')
    ffprobe_cmd.append(fileName)
    try:
        proc = await asyncio.create_subprocess_exec(*ffprobe_cmd,
                                                    stdout=asyncio.subprocess.PIPE)
        op, err = await proc.communicate()
        op = op.decode()
        for line in op.split('\n'):
            if "90" in line:
                angle = 90
            elif "180" in line:
                angle = 180
            elif "270" in line:
                angle = 270
    except OSError:
        # Likely couldn't find ffprobe
        pass

    if angle < 0:
        angle = angle + 360

    return angle


async def readFrame(videoCapture, targetFPS, rotation, mirror):
    if targetFPS > 0:
        await asyncio.sleep(1.0 / targetFPS)

    read, frame = videoCapture.read()

    if read and frame is not None and frame.size != 0:
        # Mirror frame if necessary
        if mirror:
            frame = cv2.flip(frame, 1)

        # Rotate frame if necessary
        if rotation == 90:
            frame = cv2.transpose(frame)
            frame = cv2.flip(frame, 1)
        elif rotation == 180:
            frame = cv2.flip(frame, -1)
        elif rotation == 270:
            frame = cv2.transpose(frame)
            frame = cv2.flip(frame, 0)
        else:  # 0 or some other weird result
            pass
    else:
        # This is needed because OpenCV doesn't do a good job of ensuring this.
        read = False
        frame = None

    return read, frame


# This function saves chunks to disk. Normally you would make a API call to the
# DFX Server instead.
def savePayload(chunkPayload, output):
    props = {
        "valid": chunkPayload.valid,
        "start_frame": chunkPayload.start_frame,
        "end_frame": chunkPayload.end_frame,
        "chunk_number": chunkPayload.chunk_number,
        "number_chunks": chunkPayload.number_chunks,
        "first_chunk_start_time_s": chunkPayload.first_chunk_start_time_s,
        "start_time_s": chunkPayload.start_time_s,
        "end_time_s": chunkPayload.end_time_s,
        "duration_s": chunkPayload.duration_s,
    }
    prop_path = os.path.join(output,
                             "properties{}.json".format(chunkPayload.chunk_number))
    payload_path = os.path.join(output,
                                "payload{}.bin".format(chunkPayload.chunk_number))
    meta_path = os.path.join(output, "metadata{}.bin".format(chunkPayload.chunk_number))
    try:
        with open(prop_path, "w") as f_props, open(payload_path,
                                                   "wb") as f_pay, open(meta_path,
                                                                        "wb") as f_meta:
            json.dump(props, f_props)
            f_pay.write(chunkPayload.payload_data)
            f_meta.write(chunkPayload.metadata)
    except Exception as e:
        print(e)
        return None, None, None

    return prop_path, payload_path, meta_path


# This function loads previously saved face tracking data.
# Normally, you would run a face tracker on the image
def createDFXFace(collector, jsonFace):
    face = collector.createFace(jsonFace["id"])
    face.setRect(jsonFace['rect.x'], jsonFace['rect.y'], jsonFace['rect.w'],
                 jsonFace['rect.h'])
    face.setPoseValid(jsonFace['poseValid'])
    face.setDetected(jsonFace['detected'])
    points = jsonFace['points']
    for pointId, point in points.items():
        face.addPosePoint(pointId,
                          point['x'],
                          point['y'],
                          valid=point['valid'],
                          estimated=point['estimated'],
                          quality=point['quality'])
    return face


def drawImage(dfxframe,
              renderImage,
              imageSrcName,
              frameNumber,
              videoDuration_frames,
              fps,
              isMeasurementStarted,
              results,
              message=None):
    # Render the face polygons
    for faceID in dfxframe.getFaceIdentifiers():
        for regionID in dfxframe.getRegionNames(faceID):
            if (dfxframe.getRegionIntProperty(faceID, regionID, "draw") != 0):
                polygon = dfxframe.getRegionPolygon(faceID, regionID)
                cv2.polylines(renderImage, [np.array(polygon)],
                              isClosed=True,
                              color=(255, 255, 0),
                              thickness=1,
                              lineType=cv2.LINE_AA)
    # Render the "Extracting " message
    current_row = 30
    if isMeasurementStarted:
        msg = "Extracting from {} - {} frames left ({:.2f} fps)".format(
            imageSrcName, videoDuration_frames - frameNumber, fps)
    else:
        msg = "Reading from {} - ({:.2f} fps)".format(imageSrcName, fps)
    cv2.putText(renderImage,
                msg,
                org=(10, current_row),
                fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=1,
                color=(0, 0, 0),
                thickness=1,
                lineType=cv2.LINE_AA)

    # Render the message
    if message is not None:
        current_row += 30
        cv2.putText(renderImage,
                    message,
                    org=(10, current_row),
                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=1,
                    color=(255, 0, 0),
                    thickness=1,
                    lineType=cv2.LINE_AA)

    # Render the results
    for k, v in results.items():
        current_row += 30
        cv2.putText(renderImage,
                    "{}: {}".format(k, v),
                    org=(20, current_row),
                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=1,
                    color=(0, 0, 0),
                    thickness=1,
                    lineType=cv2.LINE_AA)
    return current_row
