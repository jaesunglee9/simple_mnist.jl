# This is a cleaned version of simple_mnist.jl.
# The notebook is reactive, so I haven't yet found a way to not make a code block run 
# immediately upon startup.



using MLDatasets, JLD2
include("functions.jl")

train_x, train_y = MNIST(split=:train)[:]
train_x = reshape(train_x, :, size(train_x)[3])
test_x, test_y = MNIST(split=:test)[:]
test_x = reshape(test_x, :, size(test_x)[3])

lr = 0.3
iters = 100

gradient_descent!(network) = gradient_descent!(network, train_x, train_y, iters, lr)

if isfile("network.jl2")
    jldload("network.jl2"; network)
else
    h_layer = [32]  
    input_dim = size(train_x)[1]
    n_label = max(train_y...)+1
    network = initialize_network(input_dim, h_layer, n_label)
end
