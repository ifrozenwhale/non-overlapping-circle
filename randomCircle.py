import matplotlib.pyplot as plt
import math
import numpy as np
import random
def randomcolor():
    """
    产生随机色，随机生成16进制的颜色代码
    :return:
    """
    colorArr = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0,14)]
    return "#"+color

class Circle:
    """
    Circlel类，表示单个圆
    """
    def __init__(self, x, y, r):
        """
        :param x: x 坐标
        :param y: y 坐标
        :param r: 半径
        """
        self.x = x
        self.y = y
        self.r = r

class Grid:
    """
    网格类，用于初始化圆的位置
    在每个网格中生成一个圆，确保不会重叠圆
    """
    def __init__(self, N, MAX_STEP):
        """
        :param N: 网格个数，即初始化圆的个数
        :param MAX_STEP: 网格的边长
        """
        # print(N, MAX_STEP)
        self.grid = np.arange(0, math.sqrt(N) * MAX_STEP, MAX_STEP)
        self.gx, self.gy = np.meshgrid(self.grid, self.grid)

class Map:
    """
    图类，用于规划不等圆的排列
    """
    def __init__(self, N, R=15, r=5):
        """
        :param N: 初始化的圆个数
        :param R: 圆的最大半径
        :param r: 圆的最小半径
        """
        self.MAX_R = R
        self.MIN_R = r
        self.N = N
        # 最大步长即网格的边长，是最大半径的二倍
        self.MAX_STEP = 2 * R
        # 初始化一个圆列表
        self.circles = []
        # 表示达到局部最优后的迭代次数
        self.iter = 1
        # 初始化网格
        self.grid = Grid(N, self.MAX_STEP)
        # 初始化生成随机圆
        self.init_circle()
        # 更新圆的移动步长
        self.update_step()
        # 计算整个图的势能
        self.p = self.power()

    def init_circle(self):
        """
        初始化，生成随机圆
        :return:
        """
        # 在每个网格中生成一个半径随机的圆
        for i in range(self.N):
            x = self.grid.gx.flatten()[i]
            y = self.grid.gy.flatten()[i]
            r = random.uniform(self.MIN_R, self.MAX_R)
            c = Circle(x, y, r)
            self.add(c) # 添加到list中


    def update_step(self):
        """
        更新步长，随机步长。
        默认步长为所有圆的最小半径，有一定概率步长增加，也有一定概率步长减少。
        这样是为了尽可能避免陷入局部最优解，例如当有一圈圆形成了一堵围墙，实际上中间还可以放入圆，但是由于步长限制，无法移动。
        随机的增加步长使得跨域障碍成为可能，而随机的减少步长，使距离可能更小，更紧密
        :return:
        """
        self.step = self.extreme_r()[0]
        rd = random.random()
        if rd > 0.7:
            self.step *= (1 + random.random())
        elif rd < 0.3:
            self.step *= (1 - random.random())




    def extreme_r(self):
        """
        计算圆的最小半径和最大半径
        :return: 第一个元素为最小半径第二个为最大半径
        """
        ans = [self.MAX_R, 0]
        for i in self.circles:
            if i.r < ans[0]:
                ans[0] = i.r
            if i.r > ans[1]:
                ans[1] = i.r
        return ans

    def add(self, c):
        """
        向图中添加圆
        :param c:
        :return:
        """
        # print(c.x, c.y, c.r)
        self.circles.append(c)

    def dist(self, c1, c2):
        """
        计算两个圆之间的距离（圆心距离-半径之和）
        :param c1: 圆c1
        :param c2: 圆c2
        :return: 距离
        """
        d1 = math.sqrt((c1.x - c2.x) ** 2 + (c1.y - c2.y) ** 2)
        d2 = c1.r + c2.r
        return d1 - d2

    def power(self):
        """
        势能场，所有圆到，用距离图的中心的距离的平方的平方的和表示
        :return: 势能和
        """
        total = 0
        cx = np.mean(self.grid.grid)
        for i in self.circles:
            total += ((i.x - cx)**2 + (i.y - cx)**2)**2
        return total

    def vilid(self, c):
        """
        判断是否是有效位置。
        1. 当圆越界时，返回False
        2. 当两个圆之间的距离小于0，表示相交，无效，返回False
        :param c: 圆c
        :return:
        """
        max_v = math.sqrt(self.N) * self.MAX_STEP + self.MAX_STEP / 2
        if c.x + c.r > max_v or c.x - c.r < 0 or c.y + c.r > max_v or c.y - c.r < 0:
            return False
        for i in self.circles:
            if c != i and self.dist(c, i) < 0:
                return False
        return True

    def update_pos(self, i, dr, go=True):
        """
        更新圆的位置
        :param i: 圆i
        :param dr: 移动的距离
        :param go: True表示前进，False表示退回一步
        :return:
        """
        if go:
            i.y += dr[1]
            i.x += dr[0]
        else:
            i.x -= dr[0]
            i.y -= dr[1]

    def get_direction(self, c):
        """
        计算圆的移动方向，概率赋权
        主要概率的情况下，圆的移动方向大致朝向势能场的最低处
        次要概率情况下，移动方向完全随机
        :param c:
        :return:
        """
        direct = [1, 1]
        if(random.random() < 0.7):
            direct[0] *= (random.random()-0.5)
            direct[1] *= (random.random()-0.5)
            return direct
        cx = np.mean(self.grid.grid)
        if(c.x > cx):
            direct[0] = -1
        if(c.y > cx):
            direct[1] = -1
        # print(direct)
        return direct


    def make_closer(self, count=1):
        """
        优化算法，使得总势能最低，也即所有圆更紧密
        :param count: 迭代指数，越大，表示优化程度越高
        :return:
        """
        cnt = 0
        # 循环，当有效迭代达到上限（经过iter次计算，势能没有变化），或者计算次数上限后停止
        while self.iter < count * self.N and cnt < 1e3:
            for i in self.circles:
                self.update_step() # 更新移动步长
                dr = np.random.rand(2) * self.step / (self.iter + 0.0001)
                dr *= self.get_direction(i)
                # 随机移动
                self.update_pos(i, dr, go=True)
                # 计算当前势能场
                e = self.power()
                # 判断移动条件，概率话条件
                # 主要原则满足使得势能更低，一定概率向势能高的位置移动，这主要是考虑到为其他的圆让路，突破局部障碍
                if (e < self.p and self.vilid(i)) or (self.vilid(i) and random.random() > 0.7) :
                    self.p = e
                    self.iter = 1 # 如果有移动，则重新进行迭代计数
                    # print(e)
                else:
                    self.update_pos(i, dr, go=False)
                    self.iter += 1 # 如果停滞，迭代计数加一

            cnt += 1 # 总的计算轮数
            print(self.iter)
            # 以下为产生中间图，监督是否已经达到较优解而陷入局部障碍
            if self.iter > self.N/2:
                fig, ax = plt.subplots(subplot_kw=dict(aspect="equal"))
                ax.axis("on")
                self.plot(ax)
                ax.relim()
                ax.autoscale_view()
                for c in self.circles:
                    circ = plt.Circle((c.x, c.y), c.r, color=randomcolor(), linewidth=0, alpha=0.6)
                    ax.add_patch(circ)
                plt.savefig('img/' + str(self.iter) + '.png')
                plt.close()
        print("finish closer")


    def make_full(self, count=10):
        """
        进行图的填充，在缝隙中添加新的圆。
        1. 随机产生位置，确认圆心位置是否已被占据。
        2. 对于有效的圆心坐标，从0开始增加半径，初始步长为圆的最大半径
        3. 进行迭代计数，每次迭代，增加的半径步长降低（这里使用log2降低）
        4. 对于单个圆的插入，达到当前圆心坐标允许的最大插入半径，进行插入
        5. 如果经过cnt_plus次插入操作均失败（找不到有效位置），退出，结束优化。
        :param count:
        :return:
        """
        max_v = math.sqrt(self.N) * self.MAX_STEP
        cnt_plus = 0
        while(cnt_plus < count):
            # print(cnt_plus)
            x = random.uniform(0, max_v)
            y = random.uniform(0, max_v)
            c = Circle(x, y, 0)
            cnt = 1.01
            rd = 0
            while cnt < 400:
                rd = self.MAX_R / math.log2(cnt)
                c.r += rd
                if not self.vilid(c):
                    c.r -= rd
                    cnt += 1
            if self.vilid(c):
                cnt_plus = 0
                self.add(c)
            else:
                cnt_plus += 1

    def plot(self, ax):
        """
        最终优化结果绘图
        :param ax: matplot.axes.Axes 实例
        :return:
        """
        for c in self.circles:
            # print(c.x, c.y)
            circ = plt.Circle((c.x, c.y), c.r, color=randomcolor(), linewidth=0, alpha=0.6)
            ax.add_patch(circ)

if __name__ == '__main__':
    # 生成地图，30个圆，最大半径5，最小半径1
    m = Map(30,5,1)
    # 初始化绘图
    fig, ax = plt.subplots(subplot_kw=dict(aspect="equal"))
    ax.axis("off")
    # 降低势能
    m.make_closer()
    # 进行额外填充
    m.make_full(20)
    m.plot(ax)
    ax.relim()
    ax.autoscale_view()
    plt.savefig('img/final.png', dpi=800)
    plt.show()
