from PIL import Image
import sys
im = Image.open(sys.argv[1])
w,h = im.size
total = w * h
im = im.convert('RGB')
array = []
sum = 0
for x in range(w):
  for y in range(h):
    r,g,b = im.getpixel((x,y))
    if r==255 and g==255 and b==255:
      total = total - 1;
      continue
    sum += round(r * 0.3 + g * 0.59 + b * 0.11)
if total != 0: 
  avg = sum/total
else: 
  avg = 255
level = round((-5) * avg/255 + 5)
print("The smoke's level is : " + str(level))