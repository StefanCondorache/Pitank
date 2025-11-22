import cv2 #type: ignore
import time
import sys
from tensorflow.keras.models import load_model #type: ignore
from ImageProcessing import fix_camera_orientation, find_sign_region, preprocess_for_model, decide_action

# ==========================================
# 0. PLATFORM DETECTION (PC vs PI)
# ==========================================
IS_RASPBERRY_PI = False
try:
    import RPi.GPIO as GPIO             #type: ignore  Works only on Raspberry Pi
    from picamera2 import Picamera2     #type: ignore  Because of the above, I won't install picamera2 on my PC
    IS_RASPBERRY_PI = True
    print(">> Raspberry Pi Hardware Detected: Enaging Motors & Picamera.")
except (ImportError, RuntimeError):
    print(">> PC/Simulation Mode Detected: Using Mock GPIO & Webcam.")
    # Create Dummy Classes to prevent crashes on PC
    class MockGPIO:
        BCM = "BCM"
        OUT = "OUT"
        def setmode(self, mode): pass
        def setup(self, pin, mode): pass
        def cleanup(self): print("Mock GPIO Cleanup")
        def PWM(self, pin, freq): return MockPWM()
    
    class MockPWM:
        def start(self, duty): pass
        def ChangeDutyCycle(self, duty): pass
        def stop(self): pass
        
    GPIO = MockGPIO()

# ==========================================
# 1. HARDWARE SETUP (MOTORS)
# ==========================================
# Pin setup (BCM mode)
AIN1 = 6
AIN2 = 13
BIN1 = 19
BIN2 = 26
FREQ = 1000

GPIO.setmode(GPIO.BCM)        #type: ignore
GPIO.setup(AIN1, GPIO.OUT)    #type: ignore
GPIO.setup(AIN2, GPIO.OUT)    #type: ignore
GPIO.setup(BIN1, GPIO.OUT)    #type: ignore
GPIO.setup(BIN2, GPIO.OUT)    #type: ignore

pwm_AIN1 = GPIO.PWM(AIN1, FREQ)
pwm_AIN2 = GPIO.PWM(AIN2, FREQ)
pwm_BIN1 = GPIO.PWM(BIN1, FREQ)
pwm_BIN2 = GPIO.PWM(BIN2, FREQ)

pwm_AIN1.start(0)
pwm_AIN2.start(0)
pwm_BIN1.start(0)
pwm_BIN2.start(0)

def motor_control(pwm1, pwm2, speed):
    """Control motor with speed (-100 to 100)."""

    if not IS_RASPBERRY_PI:
        return

    if speed > 0:                   # forward
        pwm1.ChangeDutyCycle(speed)
        pwm2.ChangeDutyCycle(0)
    elif speed < 0:                 # backward
        pwm1.ChangeDutyCycle(0)
        pwm2.ChangeDutyCycle(-speed)
    else:                           # stop
        pwm1.ChangeDutyCycle(0)
        pwm2.ChangeDutyCycle(0)

def drive_tank(action_text):
    """Maps the AI decision to physical motor movements."""
    speed_a = 0
    speed_b = 0

    if action_text == "TURN LEFT":
        speed_a = -80
        speed_b = 80
    elif action_text == "TURN RIGHT":
        speed_a = 80
        speed_b = -80
    elif action_text == "TURN AROUND":
        speed_a = 100
        speed_b = -100
    elif action_text == "SEARCHING":
        speed_a = 75
        speed_b = 75
    elif action_text == "UNCERTAIN":
        speed_a = 0
        speed_b = 0

    motor_control(pwm_AIN1, pwm_AIN2, speed_a)
    motor_control(pwm_BIN1, pwm_BIN2, speed_b)

# ==========================================
# 2. CAMERA & AI SETUP
# ==========================================
picam2 = None
cap = None

if IS_RASPBERRY_PI:
    print("Initializing Picamera2...")
    picam2 = Picamera2() #type: ignore
    config = picam2.create_still_configuration()
    picam2.configure(config)
    picam2.start()
    time.sleep(2)
else:
    print("Initializing Webcam...")
    cap = cv2.VideoCapture(0) # 0 is default laptop/USB cam

print("Loading AI Model...")
try:
    model = load_model('Pitank.keras')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error: Could not load 'Pitank.keras'. Reason: {e}")
    print("Make sure you ran 'train_model.py' successfully.")
    sys.exit()

# ==========================================
# 3. MAIN LOOP
# ==========================================
print("Starting Autonomous Vehicle Loop. Press 'q' to quit.")

try:
    while True:
        # 1. Capture Frame (Hardware agnostic)
        frame = None
        if IS_RASPBERRY_PI:
            frame_rgb = picam2.capture_array() # type: ignore
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        else:
            try:    
                ret, frame = cap.read() # type: ignore
            except Exception as e:
                print(f"Webcam read error: {e}")
                break

        # 2. Processing Pipeline
        frame = fix_camera_orientation(frame)
        sign_crop = find_sign_region(frame)
        
        current_action = "SEARCHING"

        if sign_crop is not None:
            input_tensor, thresh_debug = preprocess_for_model(sign_crop, target_size=(64, 64))
            prediction = model.predict(input_tensor, verbose=0)
            current_action = decide_action(prediction[0])

            cv2.imshow("AI Vision", thresh_debug)
            cv2.imshow("Detected Crop", sign_crop)
        else:
            current_action = "SEARCHING"

        # 3. Drive Motors (or print if PC)
        if IS_RASPBERRY_PI:
            drive_tank(current_action)
        else:
            if current_action != "SEARCHING": 
                print(f"SIMULATION: Motors executing '{current_action}'")

        # 4. Display Feedback
        color = (0, 255, 0) if current_action != "SEARCHING" else (0, 255, 255)
        cv2.putText(frame, f"CMD: {current_action}", (50, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        
        display_frame = cv2.resize(frame, (640, 480))
        cv2.imshow("Rover View", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    print("\nCleaning Up...")
    pwm_AIN1.stop()
    pwm_AIN2.stop()
    pwm_BIN1.stop()
    pwm_BIN2.stop()
    GPIO.cleanup()
    
    if IS_RASPBERRY_PI:
        picam2.stop() #type: ignore
    elif cap:
        cap.release()
        
    cv2.destroyAllWindows()
    print("System Halted.")