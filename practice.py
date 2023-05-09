#  Write a program in Python to execute the Bubble sort algorithm.
# def bs(a):
# # a = name of list
#     b=len(a)-1 #nbsp; 
# # minus 1 because we always compare 2 adjacent values
#     for x in range(b):
#         for y in range(b-x):
#             a[y]=a[y+1]
#     print(a)

#     a=[32,5,3,6,7,54,87]
#     print(bs(a))

# Python program to demonstrate working
# of map. 
# Return double of n
# def addition(n):
#     return n + n
# # We double all numbers using map()
# numbers = (1, 2, 3, 4)
# result = map(addition, numbers)
# print(list(result))

# # We can also use lambda expressions with map to achieve above result.
# # Double all numbers using map and lambda

# numbers = (1, 2, 3, 4)
# result = map(lambda x: x + x, numbers)
# print(list(result))

# # Add two lists using map and lambda
  
# numbers1 = [1, 2, 3]
# numbers2 = [4, 5, 6]
  
# result = map(lambda x, y: x + y, numbers1, numbers2)
# print(list(result))

# # List of strings
# l = ['sat', 'bat', 'cat', 'mat']
  
# # map() can listify the list of strings individually
# test = list(map(list, l))
# print(test)

    
# answer=1
# number=5
# i=1
# while i <= number:
#     print("Output of ",number,"is",answer)
#     answer=answer*i
#     i+=1   

# for i in range(number):
#     answer=answer*i
    
# print(answer)

# n1= 0
# n2= 1
# n3= 0 
# count = 10
# def display(n1 + " " + n2):
#     pass

# def bar(count):
# 	if count>0:    
# 		n3 = n1 + n2
#         n1 = n2
# 		n2 = n3
# 		display(" "+n3)  
# 		bar(count-1)

# bar(count-2)


# binary search
def binary_search(list, item):
    low = 0
    high = len(list)-1
    
    while low <= high:
        mid = int((low + high) / 2)
        guess = list[mid]
        if guess == item:
            return mid
        if guess > item:
            high = mid - 1
        else:
            low = mid + 1
    return None
listSort = [3,6,9,16,25,30,45,60,75,80,90,100,132,150]
print(binary_search(listSort, 150))
print(binary_search(listSort, -1))


# Selection Sort

def findSmallest(arr):
    smallest = arr[0]
    smallest_index = 0
    for i in range(1, len(arr)):
        if arr[i] < smallest:
            smallest_index = i
    return smallest_index

def selectionSort(arr):
    newArr = []
    for i in range(len(arr)):
        smallest = findSmallest(arr)
        newArr.append(arr.pop(smallest))
    return newArr

print (selectionSort([5,9,3,8,6,2,1]))


# recursion
def look_for_key(main_box):
    pile = main_box.make_a_pile_to_look_through()
    while pile is not empty:
        box = pile.grab_a_box()
    for item in box:
        if item.is_a_box():
            pile.append(item)
        elif item.is_a_key():
            print("found the key!")

def look_for_key(box):
    for item in box:
        if item.is_a_box():
            look_for_key(item)
        elif item.is_a_key():
            print("found the key!")

def greet(name):
    print ("hello + {name}")
    greet2(name)
    print ("getting ready to say bye...")
    bye()

def greet2(name):
    print ("how are you"+ {name})

def bye():
    print ("ok bye!")

def fact(x):
    if x == 1:
        return 1
    else:
        return x * fact(x-1)

fact(3)


def quicksort(array):
    if len(array) < 2:
        return array 
    else:
        pivot = array[0] 
        less = [i for i in array[1:] if i <= pivot] 
        greater = [i for i in array[1:] if i > pivot]
        return quicksort(less) + [pivot] + quicksort(greater)

print (quicksort([10, 5, 2, 3]))

# breadth first search

def search(name):
    search_queue = deque()
    search_queue += graph[name]
    searched = []
    while search_queue:
        person = search_queue.popleft()
        if not person in searched:
            if person_is_seller(person):
                print (person + "is a mango seller!")
                return True
            else:
                search_queue += graph[person]
                searched.append(person)
        return False

search("you")


