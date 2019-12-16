import numpy

#Problem 1
arr = numpy.arange(10)
print(arr)

#Problem 2
odd_numbers = arr[1:9:2]
print(odd_numbers)

#Problem 3
arr2 = arr
arr2[arr2 % 2 != 0] = -1
print(arr2)

#Problem 4
arr2d = numpy.reshape(arr, (-1, 2))
print(arr2d)

#Problem 5
a = numpy.array([1,2,3,2,3,4,3,4,5,6])
b = numpy.array([7,2,10,2,7,4,9,4,9,8])
c = numpy.intersect1d(a, b)
print(c)

#Problem 6
a = numpy.array([1,2,3,4,5])
b = numpy.array([5,6,7,8,9])
c = numpy.intersect1d(a, b)

d = numpy.delete(a, numpy.where(a == c), axis = 0)
print(d)

#Problem 6: revised, because Rohan's solution is smarter
d = numpy.setdiff1d(a,b)
print(d)