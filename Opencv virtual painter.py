import cv2
import mediapipe as mp
import numpy as nm

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1400)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 700)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
f= cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('virtual painter.avi', f, 30, (1400,700))

mph = mp.solutions.hands
hands= mph.Hands(
    static_image_mode= False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence= 0.3,
    min_tracking_confidence= 0.8)

mpd= mp.solutions.drawing_utils
canvas = nm.zeros((700,1400,3), dtype = nm.uint8)
ppx=None
ppy=None

while True:
    success, frame = cap.read()
    if not success:
        print('error in loading cam')
        break
    fframe=cv2.flip(frame, 1)
    nframe= cv2.resize(fframe,(1400,700))
    rgb=cv2.cvtColor(nframe, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    if results.multi_hand_landmarks:
        for handlms in results.multi_hand_landmarks:
            mpd.draw_landmarks(
                nframe,
                handlms,
                mph.HAND_CONNECTIONS
            )

            h,w,c= nframe.shape
            ldm=[]
            for id, lm in enumerate(handlms.landmark):
                cx = int(lm.x*w)
                cy= int(lm.y*h)
                ldm.append([id,cx,cy])

            if ldm[8][2] < ldm[6][2] and ldm[12][2] > ldm[10][2] and ldm[16][2] > ldm[14][2] and ldm[20][2] > ldm[18][2]:
                if (ppx, ppy) == (None, None):
                    (ppx,ppy) = (ldm[8][1], ldm[8][2])
                    continue
                else:
                    cv2.line(canvas, (ppx,ppy), (ldm[8][1], ldm[8][2]), (255,255,0), 8)
                    cv2.circle(nframe, (ldm[8][1], ldm[8][2]), 10, (0,255,0), -1)
                    cv2.putText(nframe,"DRAW MODE",(20,40),cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,255),3)
                    ppx= ldm[8][1]
                    ppy=ldm[8][2]
            
            elif ldm[8][2] > ldm[6][2] and ldm[12][2] > ldm[10][2] and ldm[16][2] > ldm[14][2] and ldm[20][2] > ldm[18][2]:
                if (ppx, ppy) == (None, None):
                    (ppx,ppy) = (ldm[8][1], ldm[8][2])
                    continue
                else:
                    cv2.circle(canvas, (ldm[0][1], ldm[0][2]), 100, (0,0,0), -1)
                    cv2.circle(canvas, (ldm[1][1], ldm[1][2]), 100, (0,0,0), -1)
                    cv2.circle(canvas, (ldm[6][1], ldm[6][2]), 100, (0,0,0), -1)
                    cv2.circle(canvas, (ldm[14][1], ldm[14][2]), 100, (0,0,0), -1)
                    cv2.putText(nframe,"Eraser Mode",(20,40),cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,255),3)
                    ppx= ldm[8][1]
                    ppy=ldm[8][2]

            else:
                ppx = ldm[8][1]
                ppy = ldm[8][2]
    out.write(nframe)
    nframe = cv2.add(nframe,canvas)
    cv2.rectangle(nframe, (0, 0), (1400, 70), (30, 30, 30), -1)

    cv2.putText(
        nframe,
        "AI Virtual Painter",
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.rectangle(nframe, (0,650), (1400,700), (30,30,30), -1)

    cv2.putText(
        nframe,
        "Draw : Index Finger",
        (20,685),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,255),
        2
    )

    cv2.putText(
        nframe,
        "Erase : Fist",
        (650,685),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,255),
        2
    )

    cv2.putText(
        nframe,
        "Press q : Exit",
        (1200,685),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,255),
        2
    )

    cv2.putText(
    nframe,
    "The 98th Attempt",
    (1180,45),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.6,
    (180,180,180),
    1
    )
    cv2.imshow('Open CV Virtual Painter', nframe)
    if cv2.waitKey(1) & 0xFF== ord('q'):
        break
out.release()
cap.release()
cv2.destroyAllWindows()