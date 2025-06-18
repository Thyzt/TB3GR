import cv2
import sys
import easyocr
import numpy as np

print(cv2.__version__)
print(sys.version)


# image_path = '/Users/YunChen/Downloads/images/75834.jpg' # read image

# img = cv2.imread(image_path)

print("loading EasyOCR")
reader = easyocr.Reader(['en'], gpu = False) # instance text detector
print("EasyOCR ready")


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Webcam not detected")
    exit()


scale_factor = 0.5
ocr_interval = 5
frame_count = 0
ocr_results = []

# text = reader.readtext(img) # detect text on image




while True:
    ret, frame = cap.read()
    if not ret: 
        print("Error reading frame")
        break

    combinedtext = ''

    frame_count += 1

    small_frame = cv2.resize(frame, (0,0), fx=scale_factor, fy=scale_factor)


    # convert frame to rgb for matplotlib and easyocr

    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    if frame_count % ocr_interval == 0:

        # run ocr on resized frame

        ocr_results = reader.readtext(rgb_frame)
        print(f"OCR ran, found {len(ocr_results)} results")
        for (_, text, prob) in ocr_results:
            print(f"Detected: {text} ({prob:.2f})")


    # run OCR
    # results = reader.readtext(rgb_frame)


    # Draw boxes and text

    for (bbox, text, prob) in ocr_results:

        if prob < 0.3:
            continue

        pts = [tuple(int(x/scale_factor) for x in point) for point in bbox]

        cv2.polylines(frame, [np.array(pts)], isClosed = True, color = (0,255,0), thickness = 2)
        
        # Highlighting rectangles
        #Draw text
        cv2.putText(rgb_frame, f"{text} ({prob:.2f})", pts[0], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        combinedtext += text + " "

    print(frame_count)
    print(combinedtext)


# Shoe frame

    cv2.imshow("Text Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): break


"""
for t in text:

    print(t)

    bbox, text, score = t

    cv2.rectangle(img, bbox[0], bbox[2], (0, 255, 0), 5)

    cv2.putText(img, text, (bbox[0][0], bbox[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)


    combinedtext += text + ' '
"""

# print(combinedtext)
cap.release()
cv2.destroyAllWindows()
