# 四叶草
import turtle
import time

turtle.setup(650., 350, 200, 200)
turtle.pendown()
turtle.pensize(10)
turtle.pencolor('green')


def draw_clover(radius, rotate):  # 参数radius控制叶子的大小,rotate控制叶子的旋转
    for i in range(4):
        direction = i * 90
        turtle.seth(60 + direction + rotate)  # 控制叶子根部的角度为60度
        # turtle.fd(2*radius*pow(2,1/2)) #控制叶子根部的角度为90度
        turtle.fd(4 * radius)
        for j in range(2):
            turtle.seth(90 + direction + rotate)
            turtle.circle(radius, 180)
        turtle.seth(-60 + direction + rotate)
        turtle.fd(4 * radius)
    turtle.seth(-90)
    turtle.fd(6 * radius)


draw_clover(30, 45)
time.sleep(5)