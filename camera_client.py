import cv2
import time
import requests
from pypylon import pylon

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
BRIDGE_URL = "https://calathiform-dorsoventral-gavyn.ngrok-free.dev"
CAPTURE_INTERVAL = 2  # seconds between frames

# ─────────────────────────────────────────
# INIT BASLER CAMERA
# ─────────────────────────────────────────
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

print("📷 Basler camera started!")
print(f"🔗 Sending frames to: {BRIDGE_URL}")
print("Press Ctrl+C to stop.\n")

# ─────────────────────────────────────────
# CAPTURE & SEND LOOP
# ─────────────────────────────────────────
while camera.IsGrabbing():
    try:
        grab = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        if grab.GrabSucceeded():
            image = converter.Convert(grab)
            frame = image.GetArray()
            grab.Release()

            # Encode frame as JPEG
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

            # Send to Colab bridge
            files = {"file": ("frame.jpg", buffer.tobytes(), "image/jpeg")}
            response = requests.post(
                f"{BRIDGE_URL}/upload",
                files=files,
                timeout=5,
                headers={"ngrok-skip-browser-warning": "true"}
            )

            if response.status_code == 200:
                print(f"✅ Frame sent | {time.strftime('%H:%M:%S')}")
            else:
                print(f"⚠️  Bridge responded: {response.status_code}")

        else:
            print("❌ Grab failed, retrying...")
            grab.Release()

    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")
        break
    except Exception as e:
        print(f"❌ Error: {e}")

    time.sleep(CAPTURE_INTERVAL)

camera.StopGrabbing()
camera.Close()
print("📷 Camera closed.")
