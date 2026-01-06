import cv2
import vgamepad as vg
from pynput.keyboard import Key, Controller
import mediapipe as mp
import math

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands   # for recognition of hands
mp_draw = mp.solutions.drawing_utils # for drawing lines

gamepad = vg.VX360Gamepad()
keyboard = Controller()
# hands is for making an object to detect hands
hands = mp_hands.Hands(   
    max_num_hands = 2,
    min_detection_confidence = 0.7,
    min_tracking_confidence = 0.7
)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:

    ok, frame = cap.read()
    frame = cv2.flip(frame, 1)

    if not ok:
        break

    h, w, _ = frame.shape   # for taking the height and width of the framw
    center_x = w // 2 + 180
    center_y = h // 2 - 50

    cv2.circle(
        frame,
        (center_x, center_y),
        80,
        (255, 255, 255),
        2
    )
    cv2.circle(
        frame,
        (center_x, center_y),
        3,
        (0, 0, 255),
        -1
    )

    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks and result.multi_handedness:

        # for making same indexes of multi_hand and handedness
        for lm, info in zip(
            result.multi_hand_landmarks,
            result.multi_handedness
        ):
            mp_draw.draw_landmarks(
                frame,
                lm,
                mp_hands.HAND_CONNECTIONS
            )

            check = info.classification[0].label

            if check == "Right":

                tip = lm.landmark[9]
                point_x = int(tip.x * w)
                point_y = int(tip.y * h)
                
                dx = point_x - center_x
                dy = point_y - center_y
                dy = -dy

                dist = math.hypot(dx, dy)

                if dist > 80:
                    angle = math.atan2(dy, dx)
                    point_x = int(center_x + 80 * math.cos(angle))
                    point_y = int(center_y - 80 * math.sin(angle))

                joy_x = dx / 80
                joy_y = dy / 80

                joy_x = max(-1, min(1, joy_x))
                joy_y = max(-1, min(1, joy_y))

                gamepad.left_joystick_float(
                    joy_x, joy_y
                )
                gamepad.update()

                cv2.putText(
                    frame,
                    f"X:{joy_x: .2f}, Y:{joy_y: .2f}",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, 
                    (0, 255, 0),
                    2
                )

                cv2.circle(
                    frame,
                    (point_x, point_y),
                    10,
                    (0, 255, 0),
                    -1
                )
                cv2.line(
                    frame,
                    (center_x, center_y),
                    (point_x, point_y),
                    (181, 166, 66),
                    3
                )

            elif check == "Left":

                if lm.landmark[12].y > lm.landmark[9].y:
                    keyboard.press(Key.up)
                    keyboard.release(Key.down)
                elif lm.landmark[12].y < lm.landmark[9].y:
                    keyboard.press(Key.down)
                    keyboard.release(Key.up)
            else:
                keyboard.release(Key.down)
                keyboard.release(Key.up)


    cv2.imshow("Hands", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.deleteAllWindows()




