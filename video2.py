import cv2
import sys
import easyocr

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

ocr_interval = 5
frame_count = 0
ocr_results = []

# text = reader.readtext(img) # detect text on image

combinedtext = ''

while True:
    ret, frame = cap.read()
    if not ret: 
        print("Error reading frame")
        break

    frame_count += 1

    scale_factor = 0.5

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

        scaled_bbox = [(int(x/scale_factor), int (y/scale_factor)) for (x,y) in bbox]
        (tl, tr, br, bl) = scaled_bbox
        
        # Highlighting rectangles
        cv2.rectangle(rgb_frame, tl, br, (0, 255, 0), 2)

        #Draw text
        cv2.putText(rgb_frame, f"{text} ({prob:.2f})", (tl[0], tl[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)


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
