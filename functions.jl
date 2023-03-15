mutable struct Network
    W
    b
end

function initialize_network(input_dims, h_layers, n_labels)
    # One vector is easier to work with
    layer = copy(h_layers)
    pushfirst!(layer, input_dims)
    push!(layer, n_labels)

    W = []
    b = []
    for i in 1:length(layer)-1
        push!(W, rand(layer[i], layer[i+1])')
        push!(b, rand(1, layer[i+1])')
    end
    return Network(W, b)
end

function gradient_descent!(network, X, Y, iters, lr)
	onehot_Y = onehot(Y)
	for iter in 1:iters
		Z, A = forward(network, X)
		dW, db = backward(Z, A, network, X, onehot_Y)
		update!(network, dW, db, lr)
		if iter % 50 == 0
			println(iter)
			println(accuracy(A[end], Y))
		end
	end
	return network
end

function forward(network, X)
	n_weights = length(network.W)
	Z = Vector(undef, n_weights)
	A = Vector(undef, n_weights)

	for i in 1:n_weights
		if i == 1  # first layer. needs cleaning up
			Z[i] = network.W[i] * X .+ network.b[i]
		else
			Z[i] = network.W[i] * A[i-1] .+ network.b[i]
		end
		
		if i != n_weights  # not last layer
			A[i] = relu.(Z[i])
		else
			A[i] = softmax(Z[i], dims=1)
		end
	end
	return Z, A
end

function backward(Z, A, network, X, onehot_Y)
	n_weights = length(network.W)
	dW = Vector(undef, n_weights)
	db = Vector(undef, n_weights)
	m = size(onehot_Y)[2]  # data size

	dZ = A[end] .- onehot_Y  # loss 
	for i in n_weights:-1:1
		if i == n_weights  # last layer
			# use the initialized dZ here
			dW[i] = 1/m .* dZ * transpose(A[i-1])
			db[i] = 1/m * sum(dZ, dims=2)
		elseif i != 1  # middle layers(neither first nor last)
			dZ = transpose(network.W[i+1]) * dZ .* d_relu.(Z[i])
			dW[i] = 1/m .* dZ * transpose(A[i-1])
			db[i] = 1/m * sum(dZ, dims=2)
		else  # first layer
			dZ = transpose(network.W[i+1]) * dZ .* d_relu.(Z[i])
			dW[i] = 1/m .* dZ * transpose(X)
			db[i] = 1/m * sum(dZ, dims=2)
		end
	end
	return dW, db
end	

function update!(network, dW, db, lr)
	for i in 1:length(network.W)
		network.W[i] = network.W[i] .- (lr .* dW[i])
		network.b[i] = network.b[i] .- (lr .* db[i])
	end
end


function accuracy(A, Y)
	score = 0
	pred = map(argmax, eachcol(A))
	for i in 1:length(Y)
		if pred[i] == (Y[i] + 1)
			score += 1
		end
	end
	return score/ length(Y) * 100
end

relu(Z) = max(0, Z)
d_relu(Z) = Z > 0

function softmax(v::Vector) # to avoid NaN explosion, we use modified softmax
    max_e = max(v...)
    softmax_v = exp.(v .- max_e)
    return softmax_v ./ sum(softmax_v)
end
function softmax(A::Matrix; dims)
    n_row, n_col = size(A)
    softmax_A = Matrix{Float64}(undef, n_row, n_col)
    if dims == 1 # softmax on each column
        for i in 1:n_col
            softmax_A[:, i] = softmax(A[:, i])
        end
    end
    if dims == 2 # softmax on each row
        for i in 1:n_row
            softmax_A[i, :] = softmax(A[i, :])
        end
    end
    return softmax_A
end

function onehot(label)
	n_label = size(label)[1]
	onehotvec = zeros(10, n_label)
	for i in 1:n_label
		onehotvec[label[i]+1, i] = 1.
	end
	return onehotvec
end