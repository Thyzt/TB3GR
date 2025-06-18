import cv2
import sys
import easyocr
import matplotlib.pyplot as plt


print(cv2.__version__)
print(sys.version)


image_path = '/Users/YunChen/Downloads/images/75834.jpg' # read image

img = cv2.imread(image_path)


reader = easyocr.Reader(['en'], gpu = False) # instance text detector

text = reader.readtext(img) # detect text on image

combinedtext = ''

for t in text:

    print(t)

    bbox, text, score = t

    cv2.rectangle(img, bbox[0], bbox[2], (0, 255, 0), 5)

    cv2.putText(img, text, (bbox[0][0], bbox[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)


    combinedtext += text + ' '


print(combinedtext)
plt.imshow(img)
plt.show()