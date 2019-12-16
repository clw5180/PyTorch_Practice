
'''
问题：
现在有一个包含 N 个元素的元组或者是序列，怎样将它里面的值解压后同时赋值
给 N 个变量？
解决方案：
任何的列表、元组等序列（可迭代对象）可以通过一个简单的赋值语句解压并赋值给多
个变量。唯一的前提就是变量的数量必须跟序列元素的数量是一样的。
'''
p = (4, 5, 6)
a, _, c = p
print('a=%d, c=%d' % (a, c))

'''
问题：
如果一个可迭代对象的元素个数超过变量个数时，会抛出一个 ValueError 。那么
怎样才能从这个可迭代对象中解压出 N 个元素出来？
解决方案：
Python的 「星号表达式」 可以用来解决这个问题。比如，你在学习一门课程，在学期
末的时候，你想统计下家庭作业的平均成绩，但是排除掉第一个和最后一个分数。如果
只有四个分数，你可能就直接去简单的手动赋值，但如果有 24 个呢？这时候星号表达
式就派上用场了：
'''
grades = [50, 60, 70, 80, 90, 100, 110]
first, *middle, last = grades
print('delete first and last:', middle)  # [70, 80, 90] 注意即使是0个或1个元素，也是列表类型

def sum(items):  # 利用星号表达式实现递归，主要看思路，递归不是python擅长的
    head, *tail = items
    return head + sum(tail) if tail else head
print('sum([1,3,5,7,9]):', sum([1,3,5,7,9]))

'''
字典排序
问题：
你想创建一个字典，并且在迭代或序列化这个字典的时候能够控制元素的顺序。
解决方案：
使用 collections 模块中的OrderedDict类。比如，你想精确控制以 JSON 编码后字段的顺序，你可以先使用
OrderedDict 来构建这样的数据：
'''
import json
from collections import OrderedDict
d = OrderedDict()
d['foo'] = 1
d['bar'] = 2
d['spam'] = 3
d['grok'] = 4
for key in d:
    print(key, d[key]) # Outputs "foo 1", "bar 2", "spam 3", "grok 4"
json.dumps(d)   # json.dumps()用于将字典形式的数据转化为字符串

