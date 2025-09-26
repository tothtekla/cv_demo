import cv2
import numpy as np

cap = cv2.VideoCapture(0)

mode = 1  # 1=Cartoon, 2=Pixelate, 3=Sketch
pixel_size = 16  # also controls edge detail
max_pixel = 64
min_pixel = 2

w = 640
h = 480
aspect_ratio = w/h

print("Press 1=Pixelate, 2=Sketch, 3=Cartoon, +=increase, -=decrease detail/pixel, q=quit")

cv2.namedWindow("Interactive Filters", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Interactive Filters", cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    if mode == 3:  # Cartoon

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)  # boost contrast
        gray_blur = cv2.medianBlur(gray, 7)

        edges = cv2.Canny(gray_blur, 50, 150)
        edges = cv2.dilate(edges, None, iterations=1)
        edges = cv2.bitwise_not(edges)

        # Smooth color
        color = cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)

        # Combine
        output = cv2.bitwise_and(color, color, mask=edges)

        '''
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.medianBlur(gray, 7)
        edges = cv2.adaptiveThreshold(
            gray_blur, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, 9, 9
        )
        color = cv2.bilateralFilter(frame, 9, 300, 300)
        output = cv2.bitwise_and(color, color, mask=edges)
        
        '''

    elif mode == 1:  # Pixelate
        temp = cv2.resize(frame, (w // pixel_size, h // pixel_size), interpolation=cv2.INTER_LINEAR)
        output = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)

    elif mode == 2:  # Sketch/Edge
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (7, 7), 0)
        
        low_thresh = max(10, 50 - pixel_size*2)
        high_thresh = max(70, 150 - pixel_size*2)
        edges_canny = cv2.Canny(gray_blur, low_thresh, high_thresh)
        
        edges_adapt = cv2.adaptiveThreshold(gray_blur, 255,
                                            cv2.ADAPTIVE_THRESH_MEAN_C,
                                            cv2.THRESH_BINARY, 9, 9)
        edges = cv2.bitwise_or(edges_canny, edges_adapt)
        output = cv2.cvtColor(edges_canny, cv2.COLOR_GRAY2BGR)

    # Overlay info
    mode_name = "Cartoon" if mode==1 else "Pixelate" if mode==2 else "Sketch"
    cv2.putText(output, f"Mode: {mode} {mode_name}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(output, f"Detail/Pixel Size: {pixel_size}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Interactive Filters", output)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('1'):
        mode = 1
    elif key == ord('2'):
        mode = 2
    elif key == ord('3'):
        mode = 3
    elif key == ord('+') or key == ord('='):
        pixel_size = min(pixel_size + 2, max_pixel)
    elif key == ord('-') or key == ord('_'):
        pixel_size = max(pixel_size - 2, min_pixel)

cap.release()
cv2.destroyAllWindows()
