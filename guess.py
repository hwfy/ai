import random

rang1 = int(input("设置游戏最小值"))
rang2 = int(input("设置游戏最大值"))

num = random.randint(rang1, rang2)
guess = 'guess'
print("数字猜游戏")
i = 0
while guess != num:
    i = i + 1
    guess = int(input("请输入你猜的数字"))
    if guess == num:
        print("猜对了")
    elif guess < num:
        print("值小了")
    elif guess > num:
        print("值大了")

print("总共猜了%d" % i + "次", end='')