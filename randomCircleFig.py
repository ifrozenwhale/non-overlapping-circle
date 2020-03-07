import matplotlib.pyplot as plt
import math
import numpy as np
from matplotlib.path import Path
from PIL import Image
import cv2
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
class Border:

    def get_border_array(self, path='img/whale2.jpg', type=2):
        """
        得到边界的 array
        :param path: 图片相对路径
        :param type: 图片类型，1表示极简边界图，2表示一般的图（简笔画）
        :return: 边界数组
        """
        arr = self.get_border(path, type)
        # 计算点数
        length = 0
        for i in range(len(arr)):
            length += len(arr[i])
        x = np.zeros(length)
        y = np.zeros(length)
        cnt = 0
        for i in range(len(arr)):
            for j in range(len(arr[i])):
                x[cnt] = arr[i][j][0][0]
                y[cnt] = arr[i][j][0][1]
                cnt += 1
        # vstack就是垂直叠加组合形成一个新的数组，T是转置

        xycrop = np.vstack((x, y)).T
        return xycrop

    def get_border(self, path='img/hua.jpg', type=2):
        """
        边缘检查，得到边界数据
        :param path: 图片路径
        :param type: 图片类型，1表示极简边界图，2表示一般的图（简笔画）
        :return:
        """
        img = cv2.imread(path)  # a black objects on white image is better
        thresh = cv2.Canny(img, 128, 256)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # print(contours)
        img = np.zeros(img.shape, dtype=np.uint8)
        cv2.drawContours(img, contours, -1, (255, 0, 0), 2)  # blue
        min_side_len = img.shape[0] / 32  # 多边形边长的最小值 the minimum side length of polygon
        min_poly_len = img.shape[0] / 16  # 多边形周长的最小值 the minimum round length of polygon
        min_side_num = 3  # 多边形边数的最小值
        min_area = 30.0  # 多边形面积的最小值
        approxs = [cv2.approxPolyDP(cnt, min_side_len, True) for cnt in contours]  # 以最小边长为限制画出多边形
        approxs = [approx for approx in approxs if
                   cv2.arcLength(approx, True) > min_poly_len]  # 筛选出周长大于 min_poly_len 的多边形
        approxs = [approx for approx in approxs if len(approx) > min_side_num]  # 筛选出边长数大于 min_side_num 的多边形
        approxs = [approx for approx in approxs if cv2.contourArea(approx) > min_area]  # 筛选出面积大于 min_area_num 的多边形
        if type == 1:
            return contours
        else:
            return approxs

    def plot(self,path='img/whale2.jpg'):
        bd = self.get_border_array(path)
        # print(data)
        x = []
        y = []
        for i in bd:
            x.append(i[0])
            y.append(i[1])
        plt.scatter(x, y)
        ax = plt.gca()
        ax = ax.invert_yaxis()
        plt.show()
class Circle:
    """
    Circlel类，表示单个圆
    """
    def __init__(self, x, y, r, type=1, change=0, e=0):
        """
        :param x: x 坐标
        :param y: y 坐标
        :param r: 半径
        :param type: 圆圈的类型，1表示可移动，0表示固定（边界圆）
        :param change: 圆圈无序移动的程度
        :param e: 圆的势能
        """
        self.x = x
        self.y = y
        self.r = r
        self.type = type
        self.change = change
        self.e = 0

class Grid:
    """
    网格类，用于初始化圆的位置
    在每个网格中生成一个圆，确保不会重叠圆
    """
    def __init__(self, m, N):
        """
        :param N: 网格个数，即初始化圆的个数
        :param m: Map类的 实例
        """
        # print(N, MAX_STEP)
        max_v = np.max(m.border)
        self.grid = np.arange(0, max_v, m.MAX_STEP)
        self.gx, self.gy = np.meshgrid(self.grid, self.grid)

class Map:
    """
    图类，用于规划不等圆的排列
    """
    def __init__(self, N, R=15, r=5, path='img/whale2'):
        """
        :param N: 初始化的圆个数
        :param R: 圆的最大半径
        :param r: 圆的最小半径
        """
        self.PATH = path
        bd = Border()
        self.border = bd.get_border_array(self.PATH,1)
        self.cx = np.mean(self.border[:, 0])
        self.cy = np.mean(self.border[:, 1])
        bd.plot(path)
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
        self.grid = Grid(self, N)
        # 初始化生成随机圆
        self.init_circle()
        # 更新圆的移动步长
        self.update_step()
        # 计算整个图的势能
        self.p = self.power()

    def in_border(self, c):
        pth = Path(self.border, closed=False)
        # print(pth)
        mask = pth.contains_point([c.x, c.y])
        return mask

    def init_circle(self):
        """
        初始化，生成随机圆
        :return:
        """

        # 在边界上生成圆
        for i in self.border:
            self.add(Circle(i[0], i[1], 1,type=0))
        times = 0
        # 在每个网格中生成一个半径随机的圆
        for i in range(self.N):
            if times > 1e3: # 经过times次创建圆的操作均失败，认为达到上限
                print("max numbers " + str(i))
                break
            while(True):
                if(times > 1e3):
                    break
                gxx = self.grid.gx.flatten()
                gyy = self.grid.gy.flatten()
                xj = random.randint(0, len(gxx)-1)
                yj = random.randint(0, len(gyy)-1)
                x = gxx[xj]
                y = gyy[yj]
                r = random.uniform(self.MIN_R, self.MAX_R)
                c = Circle(x, y, r)
                if(self.vilid(c)):
                    c.e = (c.x - self.cx) ** 2 + (c.y - self.cy) ** 2
                    self.add(c) # 添加到list中
                    print("Add")
                    break
                else:
                    print(times)
                    times += 1

    def update_step(self):
        """
        更新步长，随机步长。
        默认步长为所有圆的最小半径，有一定概率步长增加，也有一定概率步长减少。
        这样是为了尽可能避免陷入局部最优解，例如当有一圈圆形成了一堵围墙，实际上中间还可以放入圆，但是由于步长限制，无法移动。
        随机的增加步长使得跨域障碍成为可能，而随机的减少步长，使距离可能更小，更紧密
        :return:
        """
        self.step = self.MIN_R
        rd = random.random()
        if rd > 0.7:
            self.step = self.MAX_R * (random.random()) * 5
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
        for i in self.circles:
            total += ((i.x - self.cx)**2 + (i.y - self.cx)**2)**2
        return total

    def vilid(self, c):
        """
        判断是否是有效位置。
        1. 当圆越界时，返回False
        2. 当两个圆之间的距离小于0，表示相交，无效，返回False
        :param c: 圆c
        :return:
        """
        # max_v = math.sqrt(self.N) * self.MAX_STEP + self.MAX_STEP / 2
        # if c.x + c.r > max_v or c.x - c.r < 0 or c.y + c.r > max_v or c.y - c.r < 0:
        #     return False
        if not self.in_border(c):
            return False
        for i in self.circles:
            if c != i and self.dist(c, i) < 0:
                # print("交")
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
        i.e = (i.x - self.cx) ** 2 + (i.y - self.cy) ** 2
    def get_direction(self, c):
        """
        计算圆的移动方向，概率赋权
        主要概率的情况下，圆的移动方向大致朝向势能场的最低处
        次要概率情况下，移动方向完全随机
        :param c:
        :return:
        """
        p = c.change
        direct = [1, 1]
        if(random.random() < p/2):
            direct[0] *= (random.random()-0.5)
            direct[1] *= (random.random()-0.5)
            return direct

        if(c.x > self.cx):
            direct[0] = -1
        if(c.y > self.cy):
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
        while self.iter < count * self.N and cnt < 500:
            for i in self.circles:
                if(i.type == 0):
                    continue
                self.update_step() # 更新移动步长
                dr = np.random.rand(2) * self.step / (self.iter + 0.0001)
                dr *= self.get_direction(i)
                # 随机移动
                old_e = i.e
                self.update_pos(i, dr, go=True)
                # 计算当前势能场
                # e = self.power()

                # 判断移动条件，概率话条件
                # 主要原则满足使得势能更低，一定概率向势能高的位置移动，这主要是考虑到为其他的圆让路，突破局部障碍
                # if (i.e < old_e and self.vilid(i)) or (self.vilid(i) and random.random() > 0.7) :
                if self.vilid(i):
                    # self.p = e
                    self.iter = 1 # 如果有移动，则重新进行迭代计数
                    # print(e)
                    i.change = 0
                else:
                    self.update_pos(i, dr, go=False)
                    self.iter += 1 # 如果停滞，迭代计数加一
                    i.change += 0.1

            cnt += 1 # 总的计算轮数
            print(self.iter)
            # 以下为产生中间图，监督是否已经达到较优解而陷入局部障碍
            if cnt % 30 == 0:
                fig, ax = plt.subplots(subplot_kw=dict(aspect="equal"))
                ax.axis("off")
                self.plot(ax)
                ax.relim()
                ax.autoscale_view()
                for c in self.circles:
                    circ = plt.Circle((c.x, c.y), c.r, color=randomcolor(), linewidth=0, alpha=0.4)
                    ax.add_patch(circ)
                plt.gca().invert_yaxis()
                plt.savefig('random/x' + str(cnt) + '.png')
                plt.close()
        print("finish closer")


    def make_full(self, count=10, R=10):
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
        max_v = np.max(self.border)
        cnt_plus = 0
        while(cnt_plus < count):
            # print(cnt_plus)
            x = random.uniform(0, max_v)
            y = random.uniform(0, max_v)
            c = Circle(x, y, 0)
            if not self.vilid(c):
                cnt_plus += 1
                print(cnt_plus)
                continue
            cnt = 1.01
            rd = 0
            while cnt < 400:
                rd = R / math.log2(cnt)
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
    m = Map(200,5,1,path='img/whale2.jpg')
    # plt.gca().invert_yaxis()
    # plt.plot(m.border[:, 0], m.border[:, 1])
    # for i in m.grid.grid:
    #     for j in m.grid.grid:
    #         print(i, j)
    #         c = Circle(i,j,r=1)
    #         if not m.vilid(c):
    #             plt.scatter(i,j,marker='^')
    # plt.show()
    # 初始化绘图
    fig, ax = plt.subplots(subplot_kw=dict(aspect="equal"))

    ax.axis("off")
    # 降低势能
    m.make_closer()
    # 进行额外填充
    m.make_full(50,R=10)
    m.plot(ax)

    ax.relim()
    ax.autoscale_view()
    plt.gca().invert_yaxis()
    plt.savefig('random/final2.png', dpi=800)
    plt.show()
