import cv2
import sys
import easyocr
import matplotlib.pyplot as plt


print(cv2.__version__)
print(sys.version)


# image_path = '/Users/YunChen/Downloads/images/75834.jpg' # read image

# img = cv2.imread(image_path)


reader = easyocr.Reader(['en'], gpu = False) # instance text detector

cap = cv2.VideoCapture(0)

if not cap.isOpened(): 
    print("Error: Webcam not detected")
    exit()

plt.ion() # Turn on interactive mode
fig, ax = plt.subplots()


# text = reader.readtext(img) # detect text on image

combinedtext = ''

while True:
    ret, frame = cap.read()
    if not ret: break

    # convert frame to rgb for matplotlib and easyocr
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    # run OCR
    results = reader.readtext(rgb_frame)


    # Draw boxes and text

    for (bbox, text, prob) in results:
        (top_left, top_right, bottom_right, bottom_left) = bbox
        top_left = tuple(map(int, top_left))
        bottom_right = tuple(map(int, bottom_right))

        # Highlighting rectangles
        cv2.rectangle(rgb_frame, top_left, bottom_right, (0, 255, 0), 2)

        #Draw text
        cv2.putText(rgb_frame, text, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5 (255, 0, 0), 2)

    ax.clear()
    ax.imshow(rgb_frame)
    ax.axis('off')
    plt.pause(0.001)

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
plt.ioff()
plt.close()