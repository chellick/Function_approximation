"""
stro = input('Enter').upper().split()
print(stro)

numers = list(map(int, input().split()))
print(numers)

numers = [int(i) for i in input().split()]
print(numers)


numers = [1, 2, 7, 14, 32, 21]

for i in numers:
    print('%', i%7)
    if i % 7:
        print(i)

num = int(input())

if num > 0:
    print('>')

elif num < 0:
    print('<')

elif num == 0:
    print('=')


string = '1' * 50 + '2' * 30

while '12' in string:
    string = string.replace('12', '', 1)
print(string)


nums = [1, 2, 6, 10, 321]
'12 26 610 10321'

for i in range(len(nums) - 1):
    print(nums[i], nums[i + 1])

def f(a, b):
    return a + b

print(f(f(1, 2), 3))


def s_square(a, b):

    return a * b  

def v_square(w, h, l):
    return s_square(w, h) * l
"""

def func(name, marks, surname=None):
    return name, surname, sum(marks) / len(marks)

def mid(*marks):
    return sum(marks) / len(marks)

# nums = [2, 2, 2, 3]
# print(mid(nums))


# print(mid(3, 4, 5))


def test(**kwargs):
    return kwargs

print(test(name='van', age=10, marks=[2, 3]))