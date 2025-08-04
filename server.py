# server_stream_mjpeg.py
import depthai as dai
import socket
import struct
import time

# TCP server setup
HOST = '0.0.0.0'
PORT = 5000
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((HOST, PORT))
server.listen(1)
print(f"Waiting for connection on {HOST}:{PORT}...")
conn, addr = server.accept()
print(f"Client connected: {addr}")

# DepthAI pipeline
pipeline = dai.Pipeline()

camRgb = pipeline.create(dai.node.ColorCamera)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
camRgb.setFps(30)

videoEnc = pipeline.create(dai.node.VideoEncoder)
videoEnc.setDefaultProfilePreset(30, dai.VideoEncoderProperties.Profile.MJPEG)

monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)
monoRight.setBoardSocket(dai.CameraBoardSocket.CAM_C)

stereo = pipeline.create(dai.node.StereoDepth)
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
stereo.setRectification(True)
stereo.useHomographyRectification(False)
stereo.setRectifyEdgeFillColor(0)
stereo.setDisparityToDepthUseSpecTranslation(False)
stereo.enableDistortionCorrection(True)

stereo.setLeftRightCheck(True)
stereo.setExtendedDisparity(True)
stereo.setSubpixel(True)
stereo.setSubpixelFractionalBits(5)
stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
stereo.setOutputSize(640, 400)

config = stereo.initialConfig.get()
config.postProcessing.speckleFilter.enable = True
config.postProcessing.speckleFilter.speckleRange = 12
config.postProcessing.temporalFilter.enable = False
config.postProcessing.temporalFilter.alpha = 0.9
config.postProcessing.temporalFilter.delta = 0
config.postProcessing.spatialFilter.enable = False
config.postProcessing.spatialFilter.holeFillingRadius = 5
config.postProcessing.spatialFilter.alpha = 0.9
config.postProcessing.spatialFilter.delta = 0
config.postProcessing.spatialFilter.numIterations = 1
config.postProcessing.thresholdFilter.minRange = 150
config.postProcessing.thresholdFilter.maxRange = 5000
config.postProcessing.decimationFilter.decimationFactor = 1
stereo.initialConfig.set(config)
stereo.initialConfig.setDisparityShift(0)
stereo.initialConfig.setMedianFilter(dai.MedianFilter.MEDIAN_OFF)
stereo.initialConfig.setConfidenceThreshold(180)

xout = pipeline.create(dai.node.XLinkOut)
xout.setStreamName("mjpeg")
camRgb.video.link(videoEnc.input)
videoEnc.bitstream.link(xout.input)

depthOut = pipeline.create(dai.node.XLinkOut)
depthOut.setStreamName("depth")
stereo.depth.link(depthOut.input)

with dai.Device(pipeline) as device:
    q_color = device.getOutputQueue("mjpeg", maxSize=1, blocking=False)
    q_depth = device.getOutputQueue("depth", maxSize=1, blocking=False)
    print("Streaming JPEG frames and depth data to client...")

    try:
        while True:
            packet_color = q_color.tryGet()
            packet_depth = q_depth.tryGet()

            color_frame = packet_color.getData() if packet_color else b''
            size_color_frame = len(color_frame) if packet_color else 0

            if packet_depth:
                depth_frame = packet_depth.getFrame()
                depth_h, depth_w = depth_frame.shape
                depth_bytes = depth_frame.tobytes()
            else:
                depth_frame = None
                depth_h, depth_w = 0, 0
                depth_bytes = b''

            # Prepare header
            tag = b'FRAME'
            padding = b'\x00' * (8 - len(tag))  # Pad tag to 8 bytes
            header = tag + padding + struct.pack('>IIII', size_color_frame, depth_w, depth_h, int(time.time()))

            print(f"Sending frame: {size_color_frame} bytes, Depth: {depth_w *depth_h * 2} bytes")

            # Send
            conn.sendall(header)
            conn.sendall(color_frame)
            conn.sendall(depth_bytes)

            time.sleep(0.03)

    except (BrokenPipeError, ConnectionResetError):
        print("Client disconnected")
    finally:
        conn.close()
        server.close()