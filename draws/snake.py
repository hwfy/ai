# 贪吃蛇
import pygame
import random
 # 初始化
pygame.init()
w = 720   #窗口宽度
h = 600   #窗口高度
ROW = 30  #行数
COL = 36  #列数
#将所有的坐标看作是一个个点，定义点类
class Point:
  row = 0
  col = 0
  def __init__(self, row, col):
    self.row = row
    self.col = col
  def copy(self):
    return Point(row = self.row,col = self.col)
#显示窗口和标题
size = (w, h)
window = pygame.display.set_mode(size)
pygame.display.set_caption('贪吃蛇')
#定义蛇头坐标
head = Point(row = ROW/2, col = COL/2)
#蛇身体
snake_list = [
  Point(row = head.row,col = head.col+1),
  Point(row = head.row,col = head.col+2),
  Point(row = head.row,col = head.col+3)
]
#产生食物
def pro_food():
  #食物不能与蛇重叠
  while True:
    pos = Point(row=random.randint(1,ROW-2), col=random.randint(1,COL-2))
    is_coll = False
    if head.row == pos.row and head.col == pos.col:
      is_coll = True
    for snake in snake_list:
      if snake.col == pos.col and snake.row == pos.row:
        is_coll = True
        break
    if not is_coll:
      return pos
food = pro_food()
#定义颜色
bg_color = (255, 255, 255)
head_color = (0, 128, 128)
food_color = (255, 255, 0)
snake_color = (200,200,200)
#给定初始方向
dire = 'left'
def rect(point, color):
  cell_width = w/COL
  cell_height = h/ROW
  left = point.col*cell_width
  top = point.row*cell_height
  pygame.draw.rect(
    window, color,
    (left,top,cell_width, cell_height, )
  )
  pass
# 游戏循环
quit = True
clock = pygame.time.Clock()
while quit:
  for event in pygame.event.get():
    #退出方式
    if event.type == pygame.QUIT:
      quit = False
    elif event.type == pygame.KEYDOWN:
      #键盘控制
      if event.key == 273 or event.key == 119:
        if dire == 'left' or dire == 'right':
          dire = 'up'
      elif event.key == 274 or event.key == 115:
        if dire == 'left' or dire == 'right':
          dire = 'down'
      elif event.key == 276 or event.key == 97:
        if dire == 'up' or dire == 'down':
          dire = 'left'
      elif event.key == 275 or event.key == 100:
        if dire == 'up' or dire == 'down':
          dire = 'right'
  #吃
  eat=(head.row == food.row and head.col == food.col)
  if eat:
    food = pro_food()
  #处理身体
  #1.原来的头换到身体最前端
  snake_list.insert(0,head.copy())
  #2.删除身体最后一个
  if not eat:
    snake_list.pop()
  #移动
  if dire == 'left':
    head.col -= 1
  elif dire == 'right':
    head.col += 1
  elif dire == 'up':
    head.row -= 1
  elif dire == 'down':
    head.row += 1
  #检测：
  dead=False
  #1.撞墙
  if head.col < 0 or head.row< 0 or head.col >= COL or head.row >= ROW:
    dead=True
  #2.撞自己
  for snake in snake_list:
    if head.col == snake.col and head.row == snake.row:
      dead=True
      break
  if dead:
    print('dead')
    quit = False
  #绘制背景
  pygame.draw.rect(window, bg_color, (0, 0, w, h))
  #蛇头
  rect(head, head_color)
  #食物
  rect(food,food_color)
  #蛇身
  for snake in snake_list:
    rect(snake,snake_color)
  pygame.display.flip()
  #游戏帧数
  clock.tick(20)