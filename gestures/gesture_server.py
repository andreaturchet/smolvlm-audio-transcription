import argparse
import asyncio
import websockets
import json
import cv2
import time
import os
import numpy as np
import base64
from main_controller import MainController
from utils import Event

# +++ADD: Configuration for target WebSocket+++
TARGET_WS_URL = "ws://localhost:9001"  # Change to your target IP:PORT
target_websocket = None  # Global variable to hold the target connection

async def connect_to_target():
    """Establish connection to target WebSocket server"""
    global target_websocket
    try:
        target_websocket = await websockets.connect(TARGET_WS_URL, max_size=10*1024*1024)
        print(f"Connected to target WebSocket: {TARGET_WS_URL}")
    except Exception as e:
        print(f"Failed to connect to target WebSocket: {e}")
        target_websocket = None

async def gesture_server(websocket, path):
    global target_websocket
    print("Gesture client connected")

    # Corrected model paths, assuming the script is run from the project root
    detector_path = "gestures/models/hand_detector.onnx"
    classifier_path = "gestures/models/crops_classifier.onnx"

    if not os.path.exists(detector_path) or not os.path.exists(classifier_path):
        print(f"Error: Model files not found. Expected at {detector_path} and {classifier_path}")
        print("Please ensure you are running the script from the project's root directory.")
        return

    try:
        controller = MainController(detector_path, classifier_path)
    except Exception as e:
        print(f"Error initializing MainController: {e}")
        print("Please ensure ONNX Runtime is installed (`pip install onnxruntime`) and model files are accessible.")
        return

    # +++ADD: Connect to target WebSocket if not connected+++
    if target_websocket is None:
        await connect_to_target()

    last_gesture_time = 0
    gesture_cooldown = 1.0  # 1-second cooldown to prevent multiple triggers

    while True:
        try:
            message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            data = json.loads(message)

            if data.get("type") != "frame":
                continue

            img_data = base64.b64decode(data["image"])
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                print("Failed to decode frame.")
                continue

        except asyncio.TimeoutError:
            continue
        except Exception as e:
            print(f"Error receiving frame: {e}")
            break

        # Process the frame to detect hands and gestures
        controller(frame)

        current_time = time.time()

        # Check for a gesture only if the cooldown period has passed
        if (current_time - last_gesture_time) > gesture_cooldown:
            if len(controller.tracks) > 0:
                for trk in controller.tracks:
                    print(trk["hands"].action)

                    # Check for a fresh action from the tracker
                    if trk["tracker"].time_since_update < 1 and trk["hands"].action is not None:
                        action = trk["hands"].action
                        action_to_send = None

                        # Map swipe gestures to commands
                        if action == Event.FAST_SWIPE_DOWN:
                            action_to_send = "next"
                        elif action == Event.FAST_SWIPE_UP:
                            action_to_send = "previous"

                        if action_to_send:
                            print(f"Sending action: {action_to_send}")

                            try:
                                if target_websocket is not None and target_websocket.open:
                                    await target_websocket.send(json.dumps({"source": "gesture", "content": action_to_send}))
                                    print(f"Sent to {TARGET_WS_URL}")
                                else:
                                    print("Target WebSocket not connected, attempting reconnect...")
                                    await connect_to_target()
                            except Exception as e:
                                print(f"Error sending to target: {e}")
                                # Try reconnecting
                                await connect_to_target()

                            # +++OPTIONAL: Also send confirmation back to frame sender+++
                            # await websocket.send(json.dumps({"type": "ack", "action": action_to_send}))

                            last_gesture_time = current_time  # Reset cooldown timer
                            trk["hands"].action = None  # Clear the action to prevent re-triggering
                            break  # Process only one gesture per frame loop

async def main():
    # Ensure the current working directory is the project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    os.chdir(project_root)
    print(f"Running server from directory: {os.getcwd()}")

    parser = argparse.ArgumentParser(description='Gesture recognition server')
    parser.add_argument('--target', type=str, default=TARGET_WS_URL,
                        help='Target WebSocket URL (e.g., ws://127.0.0.1:8080)')
    args = parser.parse_args()

    global TARGET_WS_URL
    TARGET_WS_URL = args.target
    print(f"Target WebSocket configured: {TARGET_WS_URL}")

    async with websockets.serve(gesture_server, "0.0.0.0", 9003, max_size=10*1024*1024):
        print("Gesture WebSocket server started on ws://0.0.0.0:9003")
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Server stopped manually.")
