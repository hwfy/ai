#!python

"""画一个小乌龟"""

import turtle as t;

t.pensize(2)
t.hideturtle()
t.colormode(255)
t.color((0, 0, 0), "Green")
t.setup(500, 500)
t.speed(5)

t.penup()
t.goto(0, -100)
t.pendown()
t.circle(100)

t.penup()
t.goto(-20, 35)
t.pendown()
t.begin_fill()
t.forward(40)
t.seth(-60)
t.forward(40)
t.seth(-120)
t.forward(40)
t.seth(-180)
t.forward(40)
t.seth(120)
t.forward(40)
t.seth(60)
t.forward(40)
t.end_fill()

t.seth(120)
t.color((0, 0, 0), (29, 184, 130))

for i in range(6):
    t.begin_fill()
    t.forward(60)
    t.right(90)
    t.circle(-100, 60)
    t.right(90)
    t.forward(60)
    t.right(180)
    t.end_fill()

t.penup()
t.goto(-15, 100)
t.seth(90)
t.pendown()
t.forward(15)
t.circle(-15, 180)
t.forward(15)

for i in range(4):
    t.penup()
    t.goto(0, 0)
    if i == 0:
        t.seth(35);
    if i == 1:
        t.seth(-25)
    if i == 2:
        t.seth(-145)
    if i == 3:
        t.seth(-205)
    t.forward(100)
    t.right(5)
    t.pendown()
    t.forward(10)
    t.circle(-10, 180)
    t.forward(10)

t.penup()
t.goto(10, -100)
t.seth(-90)
t.pendown()
t.forward(10)
t.circle(-30, 60)
t.right(150)
t.circle(30, 60)
t.goto(-10, -100)
