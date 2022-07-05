# 线条颜色
import turtle

t = turtle.Pen()
turtle.bgcolor("black")

sides = 5
colors = ["red", "yellow", "blue", "orange", "green", "purple"]

for x in range(360):
    t.pencolor(colors[x % sides])
    t.forward(x)
    t.left(360/sides+1)

t.width(x*sides/200)