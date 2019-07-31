import argparse
import asyncio
import json
import math
import os
import platform
import sys

import cv2
import libdfx as dfx
import numpy as np

from utils import createDFXFace, drawImage, findVideoRotation, readFrame, savePayload
from dlib_tracker import DlibTracker
from dfxapiclient import SimpleClient


class DfxExtractor():
    def __init__(self):
        self._cancel = False
        self._complete = False
        self._results = {}
        self._signal = 1

        # Create a DFX Factory object
        self._dfxFactory = dfx.Factory()
        print("Created DFX Factory:", self._dfxFactory.getVersion())

        # Create an empty self._collector
        self._collector = None

        # Create queues
        self._readerQueue = asyncio.Queue(30)
        self._chunkQueue = asyncio.Queue(5)

    @classmethod
    async def create(cls):
        """Create instance of DfxExtractor using factory.

        Needed to ensure async objects are all created in the same async loop

        Returns:
            DfxExtractor -- self

        """
        self = DfxExtractor()
        return self

    async def _readFrames(self, videoCapture, rotation, mirror, targetFPS):
        """Read frames from the videoCapture device and put into the readerQueue."""
        while not self._cancel:
            read, frame = await readFrame(videoCapture, targetFPS, rotation, mirror)
            await self._readerQueue.put((read, frame))
        await self._readerQueue.put((False, None))

    async def _processChunks(self, outputFolder):
        """Process the chunks that were added to the `_chunkQueue`."""
        while True:
            chunk = await self._chunkQueue.get()
            if chunk is None:
                break

            # Reset results
            self._results = {
                "Chunk Number": chunk.chunk_number + 1,
                "Results": "Processing..."
            }

            # Save file
            if outputFolder and outputFolder != '':
                savePayload(chunk, outputFolder)

            # Add data to measurement
            await self.dfxapiclient.add_chunk(chunk)

            self._chunkQueue.task_done()
        self._chunkQueue.task_done()

    async def _getChunk(self, collector):
        """ Get a chunk from the collector and put into the chunkQueue."""
        chunkData = collector.getChunkData()
        print("\n--------------------------------")
        if chunkData is not None:
            chunkPayload = chunkData.getChunkPayload()
            await self._chunkQueue.put(chunkPayload)
            print("Got chunk with {}".format(chunkPayload))
        else:
            print("Got empty chunk")

    async def initialize(self, studyCfgPath, dfxclient):
        """Initialize the extractor."""
        # Initialize a study
        if not self._dfxFactory.initializeStudyFromFile(studyCfgPath):
            print("DFX study initialization failed: {}".format(
                self._dfxFactory.getLastErrorMessage()))
            sys.exit(1)
        print("Created study from {}".format(studyCfgPath))

        # Create collector
        self._collector = self._dfxFactory.createCollector()
        if self._collector.getCollectorState() == dfx.CollectorState.ERROR:
            print("Collector creation failed: {}".format(
                self._collector.getLastErrorMessage()))
            sys.exit(1)
        print("Created collector")

        # Set up DFX API client
        self.dfxapiclient = dfxclient

    async def doExtraction(self,
                           imageSrcPath,
                           faceDetectStrategy,
                           outputPath=None,
                           resultsPath=None,
                           resolution=None,
                           preTrackedFacesPath=None,
                           chunkDuration=15,
                           videoDuration=60,
                           event_loop=None,
                           save_faces=False):
        """Extract TOI data from `imageSrcPath`.

        Arguments:
            imageSrcPath {String} -- Path of video file or numeric ID of web camera
            studyCfgPath {String} -- Path of study configuration file
            faceDetectStrategy {String} -- Face detection strategy ("brute", "smart" or "fast")

        Keyword Arguments:
            outputPath {String} -- Path of folder to save chunks (default: {None})
            resolution {String} -- Resolution to open camera e.g. 1280x720 (default: {None})
            preTrackedFacesPath {String} -- Path of pre-tracked face points file (default: {None})
        """
        # For saving facepoints
        facepoints = {}
        facepoints["frames"] = {}

        # Load the face tracking data
        if preTrackedFacesPath is not None:
            with open(preTrackedFacesPath, 'r') as f:
                preTrackedFaces = json.load(f)["frames"]
        else:
            preTrackedFaces = None
            tracker = DlibTracker(face_detect_strategy=faceDetectStrategy)

        # Check if camera
        try:
            imageSrcPath = int(imageSrcPath)
            isCamera = True
        except ValueError:
            isCamera = False

        # Load video capture source
        videocap = cv2.VideoCapture(imageSrcPath)
        targetFPS = videocap.get(cv2.CAP_PROP_FPS)
        try:
            durationOneFrame_ns = 1000000000.0 / targetFPS
        except ZeroDivisionError:
            raise ValueError("Invalid or nonexistent video file")

        if isCamera:
            mirror = True
            rotation = 0
            videoDuration_frames = 901
            videoFileName = "Camera {}".format(imageSrcPath)
            if resolution is not None:
                try:
                    width, height, *_ = (int(x) for x in resolution.split('x'))
                    videocap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    videocap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                except Exception as e:
                    print("Could not set resolution to {} because {}".format(
                        resolution, e))
        else:
            mirror = False
            # Probe file using ffprobe to see if it's rotated
            rotation = await findVideoRotation(imageSrcPath)

            # Count frames
            videoDuration_frames = int(1000000000.0 * int(videoDuration) /
                                       durationOneFrame_ns)
            if videoDuration_frames > int(videocap.get(cv2.CAP_PROP_FRAME_COUNT)):
                videoDuration_frames = int(videocap.get(cv2.CAP_PROP_FRAME_COUNT))
            videoFileName = os.path.basename(imageSrcPath)

        chunkDuration_s = int(chunkDuration)
        KLUDGE = 1  # This kludge to handle a bug in current SDK
        chunkFrameCount = math.ceil(chunkDuration_s * targetFPS + KLUDGE)
        numberChunks = math.ceil(videoDuration_frames / chunkFrameCount)

        # Set collector config
        self._collector.setTargetFPS(targetFPS)
        self._collector.setChunkDurationSeconds(chunkDuration_s)
        self._collector.setNumberChunks(numberChunks)

        print("    mode: {}".format(self._dfxFactory.getMode()))
        print("    number chunks: {}".format(self._collector.getNumberChunks()))
        print("    chunk duration: {}s".format(
            self._collector.getChunkDurationSeconds()))
        for constraint in self._collector.getEnabledConstraints():
            print("    enabled constraint: {}".format(constraint))

        # Create outputPath folder if it doesn't exist
        if outputPath is not None and not os.path.exists(outputPath):
            os.makedirs(outputPath)

        self._cancel = False

        # Read video frames
        asyncio.ensure_future(self._readFrames(videocap, rotation, mirror, targetFPS))
        # Add to measurement and save
        asyncio.ensure_future(self._processChunks(outputPath))
        # Subscribe to results
        asyncio.ensure_future(self.dfxapiclient.subscribe_to_results())
        # Decode and display
        asyncio.ensure_future(self.decode_results(resultsPath))

        isMeasurementStarted = False
        startFrameNumber = 0
        if not isCamera:
            # Start collection
            self._collector.startCollection()
            isMeasurementStarted = True

        # Start reading frames and adding to self._collector
        frameNumber = 0
        success = False
        while True:
            # Get frame from the readerQueue
            read, image = await self._readerQueue.get()
            # This makes sense as we are joining outside the loop
            self._readerQueue.task_done()

            if not read:
                # Video ended, so grab what should be the last, possibly truncated chunk
                await self._getChunk(self._collector)
                success = True
                break

            # Make a copy for rendering
            renderImage = np.copy(image)

            # Create a dfx_video_frame
            if isCamera:
                frameNumber += 1
            else:
                frameNumber = int(videocap.get(cv2.CAP_PROP_POS_FRAMES))

            videoFrame = dfx.VideoFrame(image, frameNumber,
                                        frameNumber * durationOneFrame_ns,
                                        dfx.ChannelOrder.CHANNEL_ORDER_BGR)

            # Create a dfx_frame from the dfx_video_frame
            frame = self._collector.createFrame(videoFrame)

            # Add the dfx_face to the dfx_frame
            if preTrackedFaces is None:
                # Track faces
                # mw = int(image.shape[1] * 0.7)
                # mh = int(image.shape[0] * 0.7)
                # mx = int((image.shape[1] - mw) * 0.5)
                # my = int((image.shape[0] - mh) * 0.5)
                faces = tracker.trackFaces(image, None)  # (mx, my, mw, mh))

                for id, jsonFace in faces.items():
                    if save_faces:
                        facepoints["frames"][frameNumber] = jsonFace
                    face = createDFXFace(self._collector, jsonFace)
                    frame.addFace(face)
            else:
                face = createDFXFace(self._collector, preTrackedFaces[str(frameNumber)])
                frame.addFace(face)

            # Add a marker to the 1000th dfx_frame
            if frameNumber == 1000:
                frame.addMarker("This is the 1000th frame")

            # Do the extraction
            self._collector.defineRegions(frame)
            result = self._collector.extractChannels(frame)

            # Grab a chunk and check if we are finished
            if result == dfx.CollectorState.CHUNKREADY or result == dfx.CollectorState.COMPLETED:
                await self._getChunk(self._collector)
                if result == dfx.CollectorState.COMPLETED:
                    print(
                        "\ndfx.CollectorState.COMPLETED at frame {}".format(frameNumber))
                    success = True
                    print("Exiting")
                    break

            # Rendering
            msg = ", press 'q' to cancel"
            if isMeasurementStarted:
                msg = "Measurement started" + msg
            else:
                msg = "Press 's' to start measurement" + msg
            renderImageLast = np.copy(renderImage)
            drawImage(frame, renderImage, videoFileName, frameNumber - startFrameNumber,
                      videoDuration_frames, targetFPS, isMeasurementStarted,
                      self._results, msg)
            cv2.imshow('DFX End-to-end Demo', renderImage)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                success = False
                break
            if not isMeasurementStarted and key == ord('s'):
                self._collector.startCollection()
                isMeasurementStarted = True
                startFrameNumber = frameNumber

        # Signal read done
        self._cancel = True

        # Flush the queue of any leftover frames
        while not self._readerQueue.empty():
            await self._readerQueue.get()
            self._readerQueue.task_done()
        await self._readerQueue.join()  # Wait for reader queue to finish

        # Send an empty frame to chunk queue to signal finish
        await self._chunkQueue.put(None)
        await self._chunkQueue.join()  # Wait for chunk queue to finish

        # Flag for signalling complete
        self._complete = True

        # Save face points if prompted
        if save_faces:
            print("\nSaving facepoints")
            imageSrcPath = imageSrcPath.strip(".mov").strip(".mp4")
            fp_file = imageSrcPath + "-faces.json"
            with open(fp_file, 'w') as fp:
                json.dump(facepoints, fp)

        msg = "Collection finished completely - press 'q' to exit" if success else "Collection interrupted or failed - press 'q' again to exit"
        print("\n", msg)
        await asyncio.sleep(self._signal)

        # Keep displaying image until dismissed
        drawImage(frame, renderImageLast, videoFileName, 0, 0, targetFPS,
                  isMeasurementStarted, self._results, msg)
        while True:
            cv2.imshow('DFX End-to-end Demo', renderImageLast)
            await asyncio.sleep(self._signal)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything done, release the capture
        videocap.release()
        cv2.destroyAllWindows()

        # Shut down DFX API client activities
        await self.dfxapiclient.shutdown()
        self.dfxapiclient.clear()

    # Decode payload results
    async def decode_results(self, outputPath):
        counter = 0
        while not self._complete:
            # Check the queue of received chunks
            if not self.dfxapiclient.received_data.empty():
                chunk = await self.dfxapiclient.received_data.get()

                # Save results if output folder given
                if outputPath and outputPath != "":
                    with open(outputPath + '/result_' + str(counter) + '.bin', 'wb') as f:
                        f.write(chunk)
                counter += 1
                print("\n Results: ")

                # Decode the data
                decoded_data = self._collector.decodeMeasurementResult(chunk)
                chunk_result = {}

                if decoded_data.getMeasurementDataKeys():
                    self._results["Results"] = "Success"  # Screen display message

                    # Iterate through keys in the measurement result
                    for key in decoded_data.getMeasurementDataKeys():
                        # Get the data for each key
                        data = decoded_data.getMeasurementData(key).getData()
                        chunk_result[key] = data

                        # Compute a mean for the data points
                        if len(data) > 0:
                            mean = round(float(sum(data) / len(data)), 2)
                            self._results[key] = mean
                        else:
                            mean = 'N/A'
                        print(key, mean)
                else:
                    print("None")
                    self._results["Results"] = "None"  # Screen display message
                self.dfxapiclient.received_data.task_done()
            else:
                await asyncio.sleep(self._signal)  # Polling


async def _setup_apiclient(license_key, study_id, email, password, chunk_length,
                           video_length, server, measurement_mode, send_method):
    # Initialize DFX API SimpleClient
    client = SimpleClient(license_key,
                          study_id,
                          email,
                          password,
                          server=server,
                          config_file="./example.config",
                          add_method=send_method,
                          chunk_length=int(chunk_length),
                          video_length=int(video_length),
                          measurement_mode=measurement_mode)

    # Create new measurement
    client.create_new_measurement()
    return client


async def main(args):
    dfxapiclient = await _setup_apiclient(args.license_key, args.study_id, args.email,
                                          args.password, args.chunklength,
                                          args.videolength, args.server,
                                          args.measurement_mode, args.send_method)
    extractor = await DfxExtractor.create()
    await extractor.initialize(studyCfgPath=args.study, dfxclient=dfxapiclient)
    await extractor.doExtraction(imageSrcPath=args.imageSrc,
                                 faceDetectStrategy=args.face_detect,
                                 outputPath=args.save_chunks_folder,
                                 resultsPath=args.save_results_folder,
                                 resolution=args.resolution,
                                 preTrackedFacesPath=args.faces,
                                 chunkDuration=args.chunklength,
                                 videoDuration=args.videolength,
                                 save_faces=args.save_facepoints)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DFX SDK Python example program")
    parser.add_argument("-v",
                        "--version",
                        action='version',
                        version='%(prog)s {}'.format(dfx.__version__))
    parser.add_argument("study", help="Path of study file")
    parser.add_argument("imageSrc",
                        help="Path of video file or numeric ID of web camera")

    parser.add_argument("license_key", help="DFX API license key")
    parser.add_argument("study_id", help="DFX API study ID")
    parser.add_argument("email", help="User email")
    parser.add_argument("password", help="User password")

    parser.add_argument("--send_method",
                        help="Method for adding/sending data to measurement",
                        choices=["REST", "rest", "websocket", "ws"],
                        default="REST")

    parser.add_argument("--measurement_mode",
                        help="Measurement mode",
                        choices=["discrete", "streaming", "batch", "video"],
                        default="discrete")

    parser.add_argument("--server",
                        help="Name of server to use",
                        choices=["qa", "dev", "prod", "prod-cn"],
                        default="qa")

    parser.add_argument(
        "--chunklength",
        help="Length of each video chunk, must be between 5 and 30 seconds",
        default=15)
    parser.add_argument("--videolength", help="Total length of video", default=60)

    parser.add_argument("-r",
                        "--resolution",
                        help="Resolution to open camera e.g. 1280x720")

    parser.add_argument("--face_detect",
                        help="Face detector caching strategy (smart by default)",
                        choices=["brute", "fast", "smart"],
                        default="smart")
    parser.add_argument("--faces", help="Path of pre-tracked face points file")

    parser.add_argument("--save_chunks_folder", help="Folder to save chunks")
    parser.add_argument("--save_results_folder", help="Folder to save results")
    parser.add_argument(
        "--save_facepoints",
        action="store_true",
        help=
        "Save the facepoints into a json file; only valid with the --face_detect brute option"
    )
    args = parser.parse_args()

    if platform.system() == 'Windows':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(args))
    loop.close()
