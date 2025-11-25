import cv2                                      #type: ignore
import time
import sys
from tensorflow.keras.models import load_model  #type: ignore

from Model.image_processing import fix_camera_orientation, find_sign_region, preprocess_for_model, decide_action

# --- GPIO SETUP (Universal) ---
IS_RASPBERRY_PI = False
try:
    import RPi.GPIO as GPIO                     #type: ignore
    from picamera2 import Picamera2             #type: ignore
    IS_RASPBERRY_PI = True
    print(">> Pi Hardware Detected.")
except:
    print(">> PC Simulation Mode.")
    class MockGPIO:
        BCM = "BCM"; OUT = "OUT"
        def setmode(self, mode): pass
        def setup(self, p, m): pass
        def cleanup(self): pass
        def PWM(self, p, f): return MockPWM()
    class MockPWM:
        def start(self, d): pass
        def ChangeDutyCycle(self, d): pass
        def stop(self): pass
    GPIO = MockGPIO()

AIN1, AIN2, BIN1, BIN2 = 6, 13, 19, 26
GPIO.setmode(GPIO.BCM)                          #type: ignore
for p in [AIN1, AIN2, BIN1, BIN2]: GPIO.setup(p, GPIO.OUT)  #type: ignore
pwm_A1 = GPIO.PWM(AIN1, 1000); pwm_A2 = GPIO.PWM(AIN2, 1000)
pwm_B1 = GPIO.PWM(BIN1, 1000); pwm_B2 = GPIO.PWM(BIN2, 1000)
for p in [pwm_A1, pwm_A2, pwm_B1, pwm_B2]: p.start(0)

def set_motor(p1, p2, speed):
    if speed > 0: p1.ChangeDutyCycle(speed); p2.ChangeDutyCycle(0)
    elif speed < 0: p1.ChangeDutyCycle(0); p2.ChangeDutyCycle(-speed)
    else: p1.ChangeDutyCycle(0); p2.ChangeDutyCycle(0)

def drive_tank(action):
    if action == "TURN LEFT": set_motor(pwm_A1, pwm_A2, -80); set_motor(pwm_B1, pwm_B2, 80)
    elif action == "TURN RIGHT": set_motor(pwm_A1, pwm_A2, 80); set_motor(pwm_B1, pwm_B2, -80)
    elif action == "TURN AROUND": set_motor(pwm_A1, pwm_A2, 100); set_motor(pwm_B1, pwm_B2, -100)
    elif action == "SEARCHING": set_motor(pwm_A1, pwm_A2, 70); set_motor(pwm_B1, pwm_B2, 70)
    else: set_motor(pwm_A1, pwm_A2, 0); set_motor(pwm_B1, pwm_B2, 0)

# --- MAIN LOOP ---
picam2 = None; cap = None
if IS_RASPBERRY_PI:
    picam2 = Picamera2()                                    #type: ignore
    picam2.configure(picam2.create_still_configuration())
    picam2.start()
    time.sleep(2)
else:
    cap = cv2.VideoCapture(0)

print("Loading Model...")
try:
    model = load_model('traffic_classifier.keras')
    print("Ready.")
except:
    print("Model not found. Run train_model.py first.")
    sys.exit()

try:
    while True:
        if IS_RASPBERRY_PI:
            frame_rgb = picam2.capture_array()                      #type: ignore   
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        else:
            ret, frame = cap.read()                                 #type: ignore
            if not ret: break

        #frame = fix_camera_orientation(frame)
        crop = find_sign_region(frame)
        
        action = "SEARCHING"
        if crop is not None:
            input_tensor, _ = preprocess_for_model(crop)
            pred = model.predict(input_tensor, verbose=0)
            action = decide_action(pred[0])
            cv2.imshow("Detected", crop)
        
        drive_tank(action)
        print(f"Action: {action}")
        
        cv2.imshow("View", cv2.resize(frame, (960, 720)))
        if cv2.waitKey(1) == ord('q'): break
finally:
    pwm_A1.stop(); pwm_A2.stop(); pwm_B1.stop(); pwm_B2.stop()
    GPIO.cleanup()
    if IS_RASPBERRY_PI: picam2.stop()                                #type: ignore
    cv2.destroyAllWindows()