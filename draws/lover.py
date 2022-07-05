# 爱心
from turtle import *
# 线条颜色
pencolor("red")
# 爱心颜色
fillcolor("pink")
# 设置一下画布的大小
setup(700, 850)
# 速度
speed(10)

# 笔的粗细
pensize(1)
# 填充实心
begin_fill()
# 角度
left(90)
# 半径 角度
circle(125, 180)
circle(360, 70)
left(38)
circle(360, 70)
circle(125, 180)

end_fill()
# done()
