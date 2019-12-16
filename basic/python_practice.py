
'''
Python中collections模块
这个模块实现了特定目标的容器，以提供Python标准内建容器 dict、list、set、tuple 的替代选择。

Counter：字典的子类，提供了可哈希对象的计数功能
defaultdict：字典的子类，提供了一个工厂函数，为字典查询提供了默认值
OrderedDict：字典的子类，保留了他们被添加的顺序
namedtuple：创建命名元组子类的工厂函数
deque：类似列表容器，实现了在两端快速添加(append)和弹出(pop)
ChainMap：类似字典的容器类，将多个映射集合到一个视图里面
详见：https://www.liaoxuefeng.com/wiki/897692888725344/973805065315456
      https://www.cnblogs.com/dianel/p/10787693.html
'''
from collections import namedtuple
Point = namedtuple('Point_2D', ['x', 'y'])
p = Point(1, 2)
print(p.x)  # 1
print(isinstance(p, Point))  # True
print(type(p)) # class '__main__.Point_2D'




