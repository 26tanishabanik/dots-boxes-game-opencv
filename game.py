import cv2
from time import sleep
import collections
from numpy.lib.shape_base import row_stack
import HandTrackingModule as htm
import time
import random
import numpy as np


def drawAll(img, buttonList = [], button1=None):
    for circle,button in zip(circleList, buttonList):
        x, y = button.pos
        w, h = button.size
        center_coordinates = circle.coor
        cv2.circle(img, center_coordinates, 90, (0,0,0), -1)
    if button1:
        button2 = button1
        x1, y1 = button2.pos
        w1, h1 = button2.size
        padding = 50
        cv2.rectangle(img, (x1 - padding, y1 - padding), (x1 + w1 + padding, y1 + h1 + padding), (175, 0, 175), cv2.FILLED)
        cv2.putText(img, button2.text, (x1 + padding, y1 + 70), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 128), 4)
    return img


class Button():
    def __init__(self, pos, row,column,text,size=[10, 10]):
        self.pos = pos
        self.size = size
        self.row = row
        self.column = column
        self.text = text
class DoneButton():
    def __init__(self, pos, text, size=[230, 230]):
        self.pos = pos
        self.size = size
        self.text = text

class Circle():
    def __init__(self, coor, text):
        self.coor = coor
        self.text = text



buttonList = []

counter = 10
num=1
for i in range(4):
    for j in range(4):
        if (i == 0 and j == 0):
            buttonList.append(Button([300 * j + 400, 50 * i + 150], str(i), str(j), str(num)))
        elif j>=0 and i != 0:
            buttonList.append(Button([300 * j + 400, 250 * i + 150], str(i), str(j), str(num)))
        else:
            buttonList.append(Button([300 * j + 400, 250 * i + 150], str(i), str(j), str(num)))
        num += 1
doneButton = DoneButton([2200, 900], "Done", size = [400, 100])

circleList = []       
for _,button in enumerate(buttonList):
    x, y = button.pos
    w, h = button.size
    center_coordinates = ((x+w//2) + 500, y+h//2)
    circleList.append(Circle(center_coordinates, button.text))

ROW_COUNT = 5
COLUMN_COUNT = 6

def create_board():
	board = np.zeros((ROW_COUNT,COLUMN_COUNT))
	return board

def drop_piece(board, row, col, piece):
	board[row][col] = piece
cap = cv2.VideoCapture(0)


show = False
detector = htm.handDetector(detectionCon=int(0.8), maxHands=1)
circles1 = []
circles2 = []
done = False
num = {}
for button in buttonList:
    num[str(button.row)+str(button.column)] = " "
board = create_board()
gameOver = False
win = -1
timer = 5
tip = False
drawColor = (255, 0, 255)
xp, yp = 0, 0
brushThickness = 15
eraserThickness = 50
imgCanvas = np.zeros((1600, 2800, 3), np.uint8)
flag = 0
lines = []
start_button = None
color_map = collections.defaultdict(int)
last_line_color = None

def add_rain(img, num_drops=100, drop_length=20):
    for _ in range(num_drops):
        x = random.randint(0, img.shape[1] - 1)
        y = random.randint(0, img.shape[0] - 1)
        color = (255, 255, 255)  # White drops for better visibility on a grayscale image
        cv2.line(img, (x, y), (x, y + drop_length), color, 2)

def check_for_boxes(lines, img, imgCanvas, detected_boxes_set, color_map, current_color):
    new_detected_boxes = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            for k in range(j + 1, len(lines)):
                for l in range(k + 1, len(lines)):
                    points = {lines[i][0], lines[i][1], lines[j][0], lines[j][1], lines[k][0], lines[k][1], lines[l][0], lines[l][1]}
                    if len(points) == 4:
                        points = list(points)
                        points.sort(key=lambda p: (p[0], p[1]))
                        if (points[0][0] == points[1][0] and points[2][0] == points[3][0] and
                                points[0][1] == points[2][1] and points[1][1] == points[3][1]):
                            box = tuple(sorted(points))  # Represent the box as a sorted tuple of points
                            if box not in detected_boxes_set:
                                detected_boxes_set.add(box)
                                new_detected_boxes.append((points, current_color))
                                if current_color == (255, 255, 0):
                                    color_map["Blue"] = color_map.get("Blue", 0) + 1
                                elif current_color == (255, 0, 255):
                                    color_map["Pink"] = color_map.get("Pink", 0) + 1
                            
    return new_detected_boxes

                           
drawing = False
end_button = None
detected_boxes_set = set()

def add_sprinkles(img, num_sprinkles=100, sprinkle_size=20):
    for _ in range(num_sprinkles):
        x = random.randint(0, img.shape[1] - 1)
        y = random.randint(0, img.shape[0] - 1)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.circle(img, (x, y), sprinkle_size, color, -1)
while True:
    success, img = cap.read()
    
    img = cv2.resize(img,(2800,1600),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)


    
    img = cv2.flip(img, 1)
   
    _, img = detector.findHands(img)
    center_x = int(img.shape[0]/2)
    center_y = int(img.shape[0]/2)
    lmList, _ = detector.findPosition(img, draw=False)
    
    

    img = drawAll(img, buttonList, doneButton)
    
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        fingers = detector.fingersUp()
        if fingers[1] and fingers[2] == False:
            if not drawing:
                for button in buttonList:
                    x, y = button.pos
                    w, h = button.size
                    cx, cy = ((x + w // 2) + 500, y + h // 2)
                    distance = ((cx - x1) ** 2 + (cy - y1) ** 2) ** 0.5
                    if distance < 30:
                        if end_button is None or end_button != button:
                            start_button = button
                            xp, yp = cx, cy
                            drawing = True
                            break
            else:
                for button in buttonList:
                    x, y = button.pos
                    w, h = button.size
                    cx, cy = ((x + w // 2) + 500, y + h // 2)
                    distance = ((cx - x1) ** 2 + (cy - y1) ** 2) ** 0.5
                    if distance < 30:
                        if start_button and start_button != button:
                            end_button = button
                            print(f"Line detected between buttons: {start_button.text} and {end_button.text}")
                            lines.append(((start_button.pos[0] + start_button.size[0] // 2,
                                        start_button.pos[1] + start_button.size[1] // 2),
                                        (end_button.pos[0] + end_button.size[0] // 2,
                                        end_button.pos[1] + end_button.size[1] // 2)))
                            # Draw proper line using OpenCV
                            start_pos = (start_button.pos[0] + start_button.size[0] // 2 + 500,
                                        start_button.pos[1] + start_button.size[1] // 2)
                            end_pos = (end_button.pos[0] + end_button.size[0] // 2 + 500,
                                    end_button.pos[1] + end_button.size[1] // 2)
                            cv2.line(img, start_pos, end_pos, drawColor, brushThickness)
                            cv2.line(imgCanvas, start_pos, end_pos, drawColor, brushThickness)
                            start_button = end_button
                            last_line_color = drawColor
                            
                            drawing = False
                        else:
                            xp, yp = x1, y1

            # check_for_boxes(lines, img, imgCanvas)
            detected_boxes = check_for_boxes(lines, img, imgCanvas, detected_boxes_set, color_map, drawColor)
        
        x3, y3 = doneButton.pos
        w3, h3 = doneButton.size
        if x3 < lmList[8][1] < x3 + w3 and y3 < lmList[8][2] < y3 + h3:
            cv2.rectangle(img, (x3 - 5, y3 - 5), (x3 + w3 + 5, y3 + h3 + 5), (193, 182, 255), cv2.FILLED)
            l, _, _ = detector.findDistance(8, 12, img)
            
            
            if l < 100:
                cv2.rectangle(img, doneButton.pos, (x1 + w3, y3 + h3), (0, 255, 0), cv2.FILLED)
                
                done = True
                if flag == 0:
                    print("done is true")
                    flag = 1
                    drawColor = (255, 255, 0)
                else:
                    flag = 0
                    drawColor = (255, 0, 255)
                if last_line_color == drawColor and drawColor == (255, 255, 0):
                    drawColor = (255, 0, 255)
                elif last_line_color == drawColor and drawColor == (255, 0, 255):
                    drawColor = (255, 255, 0)
    text1 = f"Pink Score: {color_map.get("Pink", 0)}"
    text2 = f"Blue Score: {color_map.get("Blue", 0)}"
    cv2.putText(img, text1, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (128, 0, 128), 6, cv2.LINE_AA)
    cv2.putText(img, text2, (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 6, cv2.LINE_AA)
    if len(detected_boxes_set) == 9:
        text = ""
        if color_map["Blue"] > color_map["Pink"]:
            text = "Blue wins"
        elif color_map["Blue"] < color_map["Pink"]:
            text = "Pink wins"
        else:
            text = "Draw"
        text_img = np.zeros_like(img)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 10
        font_thickness = 20
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_x = (img.shape[1] - text_size[0]) // 2
        text_y = (img.shape[0] + text_size[1]) // 2
        cv2.putText(text_img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 10, (0, 255, 0), 20, cv2.LINE_AA)
        alpha = 0.5
        cv2.addWeighted(text_img, alpha, img, 1 - alpha, 0, img)
        if text != "Draw":
            add_sprinkles(img, num_sprinkles=200, sprinkle_size=15)
        else:
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
            add_rain(img, num_drops=200, drop_length=30)     
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img,imgInv)
    img = cv2.bitwise_or(img, imgCanvas)
    cv2.imshow("Dots and Boxes", img)
    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break
    
cap.release()
cv2.destroyAllWindows()