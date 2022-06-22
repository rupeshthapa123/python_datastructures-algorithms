#  Write a program in Python to execute the Bubble sort algorithm.
def bs(a):
# a = name of list
    b=len(a)-1 #nbsp; 
# minus 1 because we always compare 2 adjacent values
    for x in range(b):
        for y in range(b-x):
            a[y]=a[y+1]
    print(a)

    a=[32,5,3,6,7,54,87]
    print(bs(a))

# Python program to demonstrate working
# of map. 
# Return double of n
def addition(n):
    return n + n
# We double all numbers using map()
numbers = (1, 2, 3, 4)
result = map(addition, numbers)
print(list(result))

# We can also use lambda expressions with map to achieve above result.
# Double all numbers using map and lambda

numbers = (1, 2, 3, 4)
result = map(lambda x: x + x, numbers)
print(list(result))

# Add two lists using map and lambda
  
numbers1 = [1, 2, 3]
numbers2 = [4, 5, 6]
  
result = map(lambda x, y: x + y, numbers1, numbers2)
print(list(result))

# List of strings
l = ['sat', 'bat', 'cat', 'mat']
  
# map() can listify the list of strings individually
test = list(map(list, l))
print(test)
