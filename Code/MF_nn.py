# @Time: 2019/1/2 0002 10:02
# @Author: KuKuXia
# Note:

"""
This script builds and runs a graph with miniflow.

There is no need to change anything to solve this quiz!

However, feel free to play with the network! Can you also
build a network that solves the equation below?

(x + y) + y
"""
"""
No need to change anything here!

If all goes well, this should work after you
modify the Add class in miniflow.py.
"""

from MF_MiniFlow import *

x, y, z, = Input(), Input(), Input()

f = Mul(x, y, z)

feed_dict = {x: 4, y: 5, z: 10}

graph = topological_sort(feed_dict)
output = forward_pass(f, graph)

# should output 19
print("{} + {} + {} = {} (according to miniflow)".format(feed_dict[x], feed_dict[y], feed_dict[z], output))

# Single example: Vector
inputs, weights, bias = Input(), Input(), Input()

f = Linear(inputs, weights, bias)

feed_dict = {
    inputs: [6, 14, 3],
    weights: [0.5, 0.25, 1.4],
    bias: 2
}

graph = topological_sort(feed_dict)
output = forward_pass(f, graph)

print(feed_dict[inputs])
value = 0
for i in range(len(feed_dict[inputs])):
    value += feed_dict[inputs][i] * feed_dict[weights][i]
print(value + feed_dict[bias])
print(output)  # should be 12.7 with this example

# Bachsize example: Matrix
X, W, b = Input(), Input(), Input()

f = Linear(X, W, b)
g = Sigmoid(f)

X_ = np.array([[-1., -2.], [-1, -2]])
W_ = np.array([[2., -3], [2., -3]])
b_ = np.array([-3., -5])

feed_dict = {X: X_, W: W_, b: b_}

graph = topological_sort(feed_dict)
output = forward_pass(g, graph)

"""
Output should be:
[[  1.23394576e-04   9.82013790e-01]
 [  1.23394576e-04   9.82013790e-01]]
"""
print(output)



"""
Test your MSE method with this script!

No changes necessary, but feel free to play
with this script to test your network.
"""
y, a = Input(), Input()
cost = MSE(y, a)

y_ = np.array([1, 2, 3])
a_ = np.array([4.5, 5, 10])

feed_dict = {y: y_, a: a_}
graph = topological_sort(feed_dict)
# forward pass
forward_pass(cost,graph)

"""
Expected output

23.4166666667
"""
print(cost.value)

