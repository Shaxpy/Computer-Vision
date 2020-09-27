import turtle
from PIL import Image
from turtle import Screen, Turtle
import cv2
polygon = Turtle()
forw = .001

turt = turtle. Turtle(visible=False)
turt.pencolor((0, 0, 100))
turt.pensize(.1)
turt.speed(0)

screen = turtle.Screen()
screen.bgcolor('black')

# turt.begin_fill()
# time.sleep(5)
# turtle.done()

while True:
    turt.forward(forw)
    turt.left(13)
    # turt.left(200)
    forw += .01
    polygon.hideturtle()
    # time.sleep(2)
    ts = screen.getcanvas().postscript(file="w.eps", colormode='color')
    inFile = Image.open("w.eps")
    outFile = "static/1.jpg"
    inFile.save(outFile)
