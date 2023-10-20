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

func relu(x):
	if x == null:
		print("Error: (relu) Invalid input to relu function")
		return
	if typeof(x) == TYPE_ARRAY:
		var result = []
		for i in range(x.size()):
			if typeof(x[i]) == TYPE_ARRAY:
				var inner_result = []
				for j in range(x[i].size()):
					if typeof(x[i][j]) != TYPE_FLOAT and typeof(x[i][j]) != TYPE_INT:
						print("Error: (relu) Array contains non-numeric value")
						print("Array: ", x)
						print("Array type of: ", typeof(x))
						return
					inner_result.append(max(0, x[i][j]))
				result.append(inner_result)
			else:
				result.append(max(0, x[i]))
		return result
	else:
		return max(0, x)

func relu_derivative(x):
	if typeof(x) == TYPE_ARRAY:
		var result = []
		for i in range(x.size()):
			if typeof(x[i]) == TYPE_ARRAY:
				var inner_result = []
				for j in range(x[i].size()):
					inner_result.append(int(x[i][j] > 0))
				result.append(inner_result)
			else:
				result.append(int(x[i] > 0))
		return result
	else:
		return int(x > 0)

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

func transpose(matrix):
	var result = []
	for i in range(matrix[0].size()):
		var row = []
		for j in range(matrix.size()):
			row.append(matrix[j][i])
		result.append(row)
	return result

func square(matrix):
	var result = []
	for i in range(len(matrix)):
		var row = []
		for j in range(len(matrix[i])):
			row.append(matrix[i][j] * matrix[i][j])
		result.append(row)
	return result

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

func fully_connected(input_size, output_size, activation):
	var weights = []
	for i in range(input_size):
		var row = []
		for j in range(output_size):
			row.append(randf() * 2 - 1)
		weights.append(row)
	return {
		'weights': weights,
		'activation': activation
	}

func train_network(network, X, y, lr, epochs, optimizer='gradient_descent'):
	var start_time = Time.get_ticks_msec()
	# Declare variables
	var beta1
	var beta2
	var epsilon
	var m = []
	var v = []
	
	# Initialize parameters for Adam optimizer
	if optimizer == 'adam':
		beta1 = 0.9
		beta2 = 0.999
		epsilon = 1e-8
		m = []
		for i in range(len(network)):
			m.append(0)

		v = []
		for i in range(len(network)):
			v.append(0)

	for epoch in range(epochs):
		# Forward propagation
		var layers = [X]
		for i in range(len(network)):
			if network[i]['activation'] == 'sigmoid':
				layers.append(sigmoid(dot(layers[i], network[i]['weights'])))
			elif network[i]['activation'] == 'relu':
				layers.append(relu(dot(layers[i], network[i]['weights'])))
				
		# Backpropagation
		var deltas = [subtract(y, layers[-1])]
		for i in range(len(network)-2, -1, -1):
			var error = dot(deltas[-1], transpose(network[i+1]['weights']))
			var delta
			if network[i]['activation'] == 'sigmoid':
				delta = matmul(error, sigmoid_derivative(layers[i+1]))
			elif network[i]['activation'] == 'relu':
				delta = error * relu_derivative(layers[i+1])
			deltas.append(delta)

		# Update weights
		for i in range(len(network)):
			if optimizer == 'gradient_descent':
				#network[i]['weights'] += lr * transpose(layers[i]).dot(deltas[-(i+1)])
				network[i]['weights'] = add(network[i]['weights'], mul_array_by_scalar(dot(transpose(layers[i]), deltas[-(i+1)]), lr))
			elif optimizer == 'adam':
#				beta1 = 0.9
#				beta2 = 0.999
#				epsilon = 1e-8
				m[i] = beta1 * m[i] + (1 - beta1) * transpose(layers[i]).dot(deltas[-(i+1)])
				v[i] = beta2 * v[i] + (1 - beta2) * square(transpose(layers[i]).dot(deltas[-(i+1)]))
				var m_hat = m[i] / (1 - pow(beta1, epoch+1))
				var v_hat = v[i] / (1 - pow(beta2, epoch+1))
				network[i]['weights'] += lr * m_hat / (sqrt(v_hat) + epsilon)
	
	var end_time = Time.get_ticks_msec()
	var elapsed_time = end_time - start_time
	print("Train time taken: " + str(elapsed_time/1000) + " s (" + str(elapsed_time) + " ms)")
	return network

func make_predictions(network, X):
	var layers = [X]
	for i in range(len(network)):
		if network[i]['activation'] == 'sigmoid':
			layers.append(sigmoid(dot(layers[i], network[i]['weights'])))
		elif network[i]['activation'] == 'relu':
			layers.append(relu(dot(layers[i], network[i]['weights'])))
	return layers[-1]

func _ready():
	# Creating the model
	var network = [fully_connected(8, 8, 'sigmoid'),
			   fully_connected(8, 8, 'sigmoid'),
			   fully_connected(8, 1, 'sigmoid')]
	
	var file = FileAccess.open("/Users/danielmilne/Documents/GitHub/GodotMLFramework/diabetes.csv", FileAccess.READ)
	var X = []
	var y = []

	if file:
		while not file.eof_reached():
			var csv_line = file.get_csv_line(",")
			var x_row = []
			for i in range(8):  # Get the first 8 columns
				x_row.append(float(csv_line[i]))
			X.append(x_row)
			y.append([float(csv_line[8])])  # Get the 9th column and make it a one-element array
		file.close()
	else:
		print("File not found")
	
	var test_size = 0.2  # The proportion of the dataset to include in the test split

	# Shuffle the indices of the data
	var indices = Array()
	for i in range(len(X)):
		indices.append(i)
	indices.shuffle()

	# Calculate the number of test samples
	var test_count = int(len(X) * test_size)

	# Split the data
	var X_train = []
	var X_test = []
	var y_train = []
	var y_test = []

	for i in range(len(indices)):
		if i < test_count:
			X_test.append(X[indices[i]])
			y_test.append(y[indices[i]])
		else:
			X_train.append(X[indices[i]])
			y_train.append(y[indices[i]])

	print("Dataset X_train size: ", len(X_train))
	print("Dataset X_test size: ", len(X_test))
	
	# Training the model
	var lr = 0.001
	var epochs = 100
	var optimizer = 'gradient_descent'
	network = train_network(network, X_train, y_train, lr, epochs, optimizer)

	# Making predictions
	var start_time = Time.get_ticks_msec()
	var predictions = make_predictions(network, X_test)

	# Create predicted_labels array
	var predicted_labels = []
	for i in range(len(predictions)):
		if predictions[i][0] > 0.5:
			predicted_labels.append(1.0)
		else:
			predicted_labels.append(0.0)

	# Calculate the number of correct predictions
	var correct_predictions = 0
	for i in range(len(predicted_labels)):
		if predicted_labels[i] == y_test[i][0]:
			correct_predictions += 1

	print("Total correct predictions: ", correct_predictions, " out of ", len(X_test))
	var end_time = Time.get_ticks_msec()
	var elapsed_time = end_time - start_time
	print("Prediction time taken: " + str(elapsed_time/1000) + " s (" + str(elapsed_time) + " ms)")
