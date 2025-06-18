import easyocr
import time

print("initializing easyocr reader")
start = time.time()
reader = easyocr.Reader(['en'])
end = time.time()

print(f"Reader initialized in {end - start:.2f} seconds.")

result = reader.readtext('75834.jpg')
print(result)