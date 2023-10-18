extends Node

func sigmoid(x):
	if x == null:
		print("Error: (sigmoid) Invalid input to sigmoid function")
		return
	if typeof(x) == TYPE_ARRAY:
		var result = []
		for i in range(x.size()):
			if typeof(x[i]) == TYPE_ARRAY:
				var inner_result = []
				for j in range(x[i].size()):
					if typeof(x[i][j]) != TYPE_FLOAT and typeof(x[i][j]) != TYPE_INT:
						print("Error: (sigmoid) Array contains non-numeric value")
						print("Array: ", x)
						print("Array type of: ", typeof(x))
						return
					inner_result.append(1.0 / (1.0 + exp(-x[i][j])))
				result.append(inner_result)
			else:
				result.append(1.0 / (1.0 + exp(-x[i])))
		return result
	else:
		return 1.0 / (1.0 + exp(-x))

func sigmoid_derivative(x):
	if typeof(x) == TYPE_ARRAY:
		var result = []
		for i in range(x.size()):
			if typeof(x[i]) == TYPE_ARRAY:
				var inner_result = []
				for j in range(x[i].size()):
					inner_result.append(x[i][j] * (1.0 - x[i][j]))
				result.append(inner_result)
			else:
				result.append(x[i] * (1.0 - x[i]))
		return result
	else:
		return x * (1.0 - x)

func add(array1, array2):
	if array1.size() != array2.size():
		print("Error: Arrays have different sizes")
		return null
	var result = []
	for i in range(array1.size()):
		if typeof(array1[i]) == TYPE_ARRAY and typeof(array2[i]) == TYPE_ARRAY:
			var inner_result = []
			for j in range(array1[i].size()):
				inner_result.append(array1[i][j] + array2[i][j])
			result.append(inner_result)
		else:
			result.append(array1[i] + array2[i])
	return result

func subtract(array1, array2):
	if array1.size() != array2.size():
		print("Error: Arrays have different sizes")
		return null
	var result = []
	for i in range(array1.size()):
		if typeof(array1[i]) == TYPE_ARRAY and typeof(array2[i]) == TYPE_ARRAY:
			var inner_result = []
			for j in range(array1[i].size()):
				inner_result.append(array1[i][j] - array2[i][j])
			result.append(inner_result)
		else:
			result.append(array1[i] - array2[i])
	return result

func matmul(array1, array2):
	if array1.size() != array2.size():
		print("Error: Arrays have different sizes")
		return null
	var result = []
	for i in range(array1.size()):
		if typeof(array1[i]) == TYPE_ARRAY and typeof(array2[i]) == TYPE_ARRAY:
			var inner_result = []
			for j in range(array1[i].size()):
				inner_result.append(array1[i][j] * array2[i][j])
			result.append(inner_result)
		else:
			result.append(array1[i] * array2[i])
	return result

func dot(matrix1, matrix2):
	if matrix1 == null or matrix2 == null:
		print("Error: (dot) One or both matrices are null")
		return
	if typeof(matrix1[0]) != TYPE_ARRAY or typeof(matrix2) != TYPE_ARRAY:
		print("Matrix1: ", matrix1[0])
		print("Matrix2: ", matrix2)
		print("Error: (dot) One or both inputs are not arrays")
		return
	if matrix1[0].size() != matrix2.size():
		print("Error: (dot) Matrices have incompatible sizes")
		return
	var result = []
	for i in range(matrix1.size()):
		result.append([])
		for j in range(matrix2[0].size()):
			var sum = 0
			for k in range(matrix1[0].size()):
				sum += matrix1[i][k] * matrix2[k][j]
			result[i].append(sum)
	return result

func mul_array_by_scalar(array, scalar):
	var result = []
	for i in range(array.size()):
		if typeof(array[i]) == TYPE_ARRAY:
			var inner_result = []
			for j in range(array[i].size()):
				inner_result.append(array[i][j] * scalar)
			result.append(inner_result)
		else:
			result.append(array[i] * scalar)
	return result

func transpose(matrix):
	var result = []
	for i in range(matrix[0].size()):
		var row = []
		for j in range(matrix.size()):
			row.append(matrix[j][i])
		result.append(row)
	return result

func initialize_weights(input_sz, hidden_sz, output_sz):
	var weights_input_hidden = []
	var weights_hidden_output = []

	# Initialize weights_input_hidden
	for i in range(input_sz):
		weights_input_hidden.append([])
		for j in range(hidden_sz):
			weights_input_hidden[i].append(randf())

	# Initialize weights_hidden_output
	for i in range(hidden_sz):
		weights_hidden_output.append([])
		for j in range(output_sz):
			weights_hidden_output[i].append(randf())

	return [weights_input_hidden, weights_hidden_output]

func feedforward(X1, weights_input_hidden1, weights_hidden_output1):
	var dp_result = dot(X1, weights_input_hidden1)
	if dp_result == null:
		print("Error: dot returned null")
		return
	var hidden_output = sigmoid(dp_result)
	if hidden_output == null:
		print("Error: sigmoid returned null")
		return
	# Calculate output
	var output = sigmoid(dot(hidden_output, weights_hidden_output1))
	return [hidden_output, output]

func optim_sgd(X1, y1, learning_rate1, weights_input_hidden1, weights_hidden_output1, hidden_output, output):
	if y1 == null or output == null:
		print("Error: y1 or output is null")
		return
	# Calculate output error and output delta
	var output_error = subtract(y1, output)
	var output_delta = matmul(output_error, sigmoid_derivative(output))

	# Calculate hidden layer error and delta
	var hidden_error = dot(output_delta, transpose(weights_hidden_output1))
	var hidden_delta = matmul(hidden_error, sigmoid_derivative(hidden_output))

	# Update weights
	weights_hidden_output1 = add(weights_hidden_output1, mul_array_by_scalar(dot(transpose(hidden_output), output_delta), learning_rate1))
	weights_input_hidden1 = add(weights_input_hidden1, mul_array_by_scalar(dot(transpose(X1), hidden_delta), learning_rate1))


	return [weights_input_hidden1, weights_hidden_output1]

func _ready():
	var X = [[0, 0], [0, 1], [1, 0], [1, 1]]
	var y = [[0], [1], [1], [0]]
	
	
	print("Dataset X size: ", len(X))
	print("Dataset X size: ", len(X[0]))
	print("Dataset y size: ", len(y))
	var input_size = 2
	var hidden_size = 16
	var output_size = 1
	
	# Initialize the weights
	var result = initialize_weights(input_size, hidden_size, output_size)
	var weights_input_hidden = result[0]
	var weights_hidden_output = result[1]
	
	# Train the neural network
	var epochs = 1000
	var learning_rate = 0.1
	
	for epoch in range(epochs):
		var result_ff = feedforward(X, weights_input_hidden, weights_hidden_output)
		var hidden_output = result_ff[0]
		var output_ff = result_ff[1]
		
		var result_bp = optim_sgd(X, y, learning_rate, weights_input_hidden, weights_hidden_output, hidden_output, output_ff)
		weights_input_hidden = result_bp[0]
		weights_hidden_output = result_bp[1]

	# Test the trained network
	var test_data = [[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]]
	for data in test_data:
		var pred1 = feedforward(data, weights_input_hidden, weights_hidden_output)
		var prediction = pred1[1]
		print("Input: ", data)
		print("Output: ", prediction)
