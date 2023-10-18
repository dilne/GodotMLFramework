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
	#Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome
#	var X = [[6,148,72,35,0,33.6,0.627,50],
#	[1,85,66,29,0,26.6,0.351,31]]
	
	
	var X = [[6.0, 148.0, 72.0, 35.0, 0.0, 33.6, 0.627, 50.0],
[1.0, 85.0, 66.0, 29.0, 0.0, 26.6, 0.351, 31.0],
[8.0, 183.0, 64.0, 0.0, 0.0, 23.3, 0.672, 32.0],
[1.0, 89.0, 66.0, 23.0, 94.0, 28.1, 0.167, 21.0],
[0.0, 137.0, 40.0, 35.0, 168.0, 43.1, 2.288, 33.0],
[5.0, 116.0, 74.0, 0.0, 0.0, 25.6, 0.201, 30.0],
[3.0, 78.0, 50.0, 32.0, 88.0, 31.0, 0.248, 26.0],
[10.0, 115.0, 0.0, 0.0, 0.0, 35.3, 0.134, 29.0],
[2.0, 197.0, 70.0, 45.0, 543.0, 30.5, 0.158, 53.0],
[8.0, 125.0, 96.0, 0.0, 0.0, 0.0, 0.232, 54.0],
[4.0, 110.0, 92.0, 0.0, 0.0, 37.6, 0.191, 30.0],
[10.0, 168.0, 74.0, 0.0, 0.0, 38.0, 0.537, 34.0],
[10.0, 139.0, 80.0, 0.0, 0.0, 27.1, 1.441, 57.0],
[1.0, 189.0, 60.0, 23.0, 846.0, 30.1, 0.398, 59.0],
[5.0, 166.0, 72.0, 19.0, 175.0, 25.8, 0.587, 51.0],
[7.0, 100.0, 0.0, 0.0, 0.0, 30.0, 0.484, 32.0],
[0.0, 118.0, 84.0, 47.0, 230.0, 45.8, 0.551, 31.0],
[7.0, 107.0, 74.0, 0.0, 0.0, 29.6, 0.254, 31.0],
[1.0, 103.0, 30.0, 38.0, 83.0, 43.3, 0.183, 33.0],
[1.0, 115.0, 70.0, 30.0, 96.0, 34.6, 0.529, 32.0],
[3.0, 126.0, 88.0, 41.0, 235.0, 39.3, 0.704, 27.0],
[8.0, 99.0, 84.0, 0.0, 0.0, 35.4, 0.388, 50.0],
[7.0, 196.0, 90.0, 0.0, 0.0, 39.8, 0.451, 41.0],
[9.0, 119.0, 80.0, 35.0, 0.0, 29.0, 0.263, 29.0],
[11.0, 143.0, 94.0, 33.0, 146.0, 36.6, 0.254, 51.0],
[10.0, 125.0, 70.0, 26.0, 115.0, 31.1, 0.205, 41.0],
[7.0, 147.0, 76.0, 0.0, 0.0, 39.4, 0.257, 43.0],
[1.0, 97.0, 66.0, 15.0, 140.0, 23.2, 0.487, 22.0],
[13.0, 145.0, 82.0, 19.0, 110.0, 22.2, 0.245, 57.0],
[5.0, 117.0, 92.0, 0.0, 0.0, 34.1, 0.337, 38.0],
[5.0, 109.0, 75.0, 26.0, 0.0, 36.0, 0.546, 60.0],
[3.0, 158.0, 76.0, 36.0, 245.0, 31.6, 0.851, 28.0],
[3.0, 88.0, 58.0, 11.0, 54.0, 24.8, 0.267, 22.0],
[6.0, 92.0, 92.0, 0.0, 0.0, 19.9, 0.188, 28.0],
[10.0, 122.0, 78.0, 31.0, 0.0, 27.6, 0.512, 45.0],
[4.0, 103.0, 60.0, 33.0, 192.0, 24.0, 0.966, 33.0],
[11.0, 138.0, 76.0, 0.0, 0.0, 33.2, 0.42, 35.0],
[9.0, 102.0, 76.0, 37.0, 0.0, 32.9, 0.665, 46.0],
[2.0, 90.0, 68.0, 42.0, 0.0, 38.2, 0.503, 27.0],
[4.0, 111.0, 72.0, 47.0, 207.0, 37.1, 1.39, 56.0],
[3.0, 180.0, 64.0, 25.0, 70.0, 34.0, 0.271, 26.0],
[7.0, 133.0, 84.0, 0.0, 0.0, 40.2, 0.696, 37.0],
[7.0, 106.0, 92.0, 18.0, 0.0, 22.7, 0.235, 48.0],
[9.0, 171.0, 110.0, 24.0, 240.0, 45.4, 0.721, 54.0],
[7.0, 159.0, 64.0, 0.0, 0.0, 27.4, 0.294, 40.0],
[0.0, 180.0, 66.0, 39.0, 0.0, 42.0, 1.893, 25.0],
[1.0, 146.0, 56.0, 0.0, 0.0, 29.7, 0.564, 29.0],
[2.0, 71.0, 70.0, 27.0, 0.0, 28.0, 0.586, 22.0],
[7.0, 103.0, 66.0, 32.0, 0.0, 39.1, 0.344, 31.0],
[7.0, 105.0, 0.0, 0.0, 0.0, 0.0, 0.305, 24.0],
[1.0, 103.0, 80.0, 11.0, 82.0, 19.4, 0.491, 22.0],
[1.0, 101.0, 50.0, 15.0, 36.0, 24.2, 0.526, 26.0],
[5.0, 88.0, 66.0, 21.0, 23.0, 24.4, 0.342, 30.0],
[8.0, 176.0, 90.0, 34.0, 300.0, 33.7, 0.467, 58.0],
[7.0, 150.0, 66.0, 42.0, 342.0, 34.7, 0.718, 42.0],
[1.0, 73.0, 50.0, 10.0, 0.0, 23.0, 0.248, 21.0],
[7.0, 187.0, 68.0, 39.0, 304.0, 37.7, 0.254, 41.0],
[0.0, 100.0, 88.0, 60.0, 110.0, 46.8, 0.962, 31.0],
[0.0, 146.0, 82.0, 0.0, 0.0, 40.5, 1.781, 44.0],
[0.0, 105.0, 64.0, 41.0, 142.0, 41.5, 0.173, 22.0],
[2.0, 84.0, 0.0, 0.0, 0.0, 0.0, 0.304, 21.0],
[8.0, 133.0, 72.0, 0.0, 0.0, 32.9, 0.27, 39.0],
[5.0, 44.0, 62.0, 0.0, 0.0, 25.0, 0.587, 36.0],
[2.0, 141.0, 58.0, 34.0, 128.0, 25.4, 0.699, 24.0],
[7.0, 114.0, 66.0, 0.0, 0.0, 32.8, 0.258, 42.0],
[5.0, 99.0, 74.0, 27.0, 0.0, 29.0, 0.203, 32.0],
[0.0, 109.0, 88.0, 30.0, 0.0, 32.5, 0.855, 38.0],
[2.0, 109.0, 92.0, 0.0, 0.0, 42.7, 0.845, 54.0],
[1.0, 95.0, 66.0, 13.0, 38.0, 19.6, 0.334, 25.0],
[4.0, 146.0, 85.0, 27.0, 100.0, 28.9, 0.189, 27.0],
[2.0, 100.0, 66.0, 20.0, 90.0, 32.9, 0.867, 28.0],
[5.0, 139.0, 64.0, 35.0, 140.0, 28.6, 0.411, 26.0],
[13.0, 126.0, 90.0, 0.0, 0.0, 43.4, 0.583, 42.0],
[4.0, 129.0, 86.0, 20.0, 270.0, 35.1, 0.231, 23.0],
[1.0, 79.0, 75.0, 30.0, 0.0, 32.0, 0.396, 22.0],
[1.0, 0.0, 48.0, 20.0, 0.0, 24.7, 0.14, 22.0],
[7.0, 62.0, 78.0, 0.0, 0.0, 32.6, 0.391, 41.0],
[5.0, 95.0, 72.0, 33.0, 0.0, 37.7, 0.37, 27.0],
[0.0, 131.0, 0.0, 0.0, 0.0, 43.2, 0.27, 26.0],
[2.0, 112.0, 66.0, 22.0, 0.0, 25.0, 0.307, 24.0],
[3.0, 113.0, 44.0, 13.0, 0.0, 22.4, 0.14, 22.0],
[2.0, 74.0, 0.0, 0.0, 0.0, 0.0, 0.102, 22.0],
[7.0, 83.0, 78.0, 26.0, 71.0, 29.3, 0.767, 36.0],
[0.0, 101.0, 65.0, 28.0, 0.0, 24.6, 0.237, 22.0],
[5.0, 137.0, 108.0, 0.0, 0.0, 48.8, 0.227, 37.0],
[2.0, 110.0, 74.0, 29.0, 125.0, 32.4, 0.698, 27.0],
[13.0, 106.0, 72.0, 54.0, 0.0, 36.6, 0.178, 45.0],
[2.0, 100.0, 68.0, 25.0, 71.0, 38.5, 0.324, 26.0],
[15.0, 136.0, 70.0, 32.0, 110.0, 37.1, 0.153, 43.0],
[1.0, 107.0, 68.0, 19.0, 0.0, 26.5, 0.165, 24.0],
[1.0, 80.0, 55.0, 0.0, 0.0, 19.1, 0.258, 21.0],
[4.0, 123.0, 80.0, 15.0, 176.0, 32.0, 0.443, 34.0],
[7.0, 81.0, 78.0, 40.0, 48.0, 46.7, 0.261, 42.0],
[4.0, 134.0, 72.0, 0.0, 0.0, 23.8, 0.277, 60.0],
[2.0, 142.0, 82.0, 18.0, 64.0, 24.7, 0.761, 21.0],
[6.0, 144.0, 72.0, 27.0, 228.0, 33.9, 0.255, 40.0],
[2.0, 92.0, 62.0, 28.0, 0.0, 31.6, 0.13, 24.0],
[1.0, 71.0, 48.0, 18.0, 76.0, 20.4, 0.323, 22.0],
[6.0, 93.0, 50.0, 30.0, 64.0, 28.7, 0.356, 23.0],
[1.0, 122.0, 90.0, 51.0, 220.0, 49.7, 0.325, 31.0],
[1.0, 163.0, 72.0, 0.0, 0.0, 39.0, 1.222, 33.0],
[1.0, 151.0, 60.0, 0.0, 0.0, 26.1, 0.179, 22.0],
[0.0, 125.0, 96.0, 0.0, 0.0, 22.5, 0.262, 21.0],
[1.0, 81.0, 72.0, 18.0, 40.0, 26.6, 0.283, 24.0],
[2.0, 85.0, 65.0, 0.0, 0.0, 39.6, 0.93, 27.0],
[1.0, 126.0, 56.0, 29.0, 152.0, 28.7, 0.801, 21.0],
[1.0, 96.0, 122.0, 0.0, 0.0, 22.4, 0.207, 27.0],
[4.0, 144.0, 58.0, 28.0, 140.0, 29.5, 0.287, 37.0],
[3.0, 83.0, 58.0, 31.0, 18.0, 34.3, 0.336, 25.0],
[0.0, 95.0, 85.0, 25.0, 36.0, 37.4, 0.247, 24.0],
[3.0, 171.0, 72.0, 33.0, 135.0, 33.3, 0.199, 24.0],
[8.0, 155.0, 62.0, 26.0, 495.0, 34.0, 0.543, 46.0],
[1.0, 89.0, 76.0, 34.0, 37.0, 31.2, 0.192, 23.0],
[4.0, 76.0, 62.0, 0.0, 0.0, 34.0, 0.391, 25.0],
[7.0, 160.0, 54.0, 32.0, 175.0, 30.5, 0.588, 39.0],
[4.0, 146.0, 92.0, 0.0, 0.0, 31.2, 0.539, 61.0],
[5.0, 124.0, 74.0, 0.0, 0.0, 34.0, 0.22, 38.0],
[5.0, 78.0, 48.0, 0.0, 0.0, 33.7, 0.654, 25.0],
[4.0, 97.0, 60.0, 23.0, 0.0, 28.2, 0.443, 22.0],
[4.0, 99.0, 76.0, 15.0, 51.0, 23.2, 0.223, 21.0],
[0.0, 162.0, 76.0, 56.0, 100.0, 53.2, 0.759, 25.0],
[6.0, 111.0, 64.0, 39.0, 0.0, 34.2, 0.26, 24.0],
[2.0, 107.0, 74.0, 30.0, 100.0, 33.6, 0.404, 23.0],
[5.0, 132.0, 80.0, 0.0, 0.0, 26.8, 0.186, 69.0],
[0.0, 113.0, 76.0, 0.0, 0.0, 33.3, 0.278, 23.0],
[1.0, 88.0, 30.0, 42.0, 99.0, 55.0, 0.496, 26.0],
[3.0, 120.0, 70.0, 30.0, 135.0, 42.9, 0.452, 30.0],
[1.0, 118.0, 58.0, 36.0, 94.0, 33.3, 0.261, 23.0],
[1.0, 117.0, 88.0, 24.0, 145.0, 34.5, 0.403, 40.0],
[0.0, 105.0, 84.0, 0.0, 0.0, 27.9, 0.741, 62.0],
[4.0, 173.0, 70.0, 14.0, 168.0, 29.7, 0.361, 33.0],
[9.0, 122.0, 56.0, 0.0, 0.0, 33.3, 1.114, 33.0],
[3.0, 170.0, 64.0, 37.0, 225.0, 34.5, 0.356, 30.0],
[8.0, 84.0, 74.0, 31.0, 0.0, 38.3, 0.457, 39.0],
[2.0, 96.0, 68.0, 13.0, 49.0, 21.1, 0.647, 26.0],
[2.0, 125.0, 60.0, 20.0, 140.0, 33.8, 0.088, 31.0],
[0.0, 100.0, 70.0, 26.0, 50.0, 30.8, 0.597, 21.0],
[0.0, 93.0, 60.0, 25.0, 92.0, 28.7, 0.532, 22.0],
[0.0, 129.0, 80.0, 0.0, 0.0, 31.2, 0.703, 29.0],
[5.0, 105.0, 72.0, 29.0, 325.0, 36.9, 0.159, 28.0],
[3.0, 128.0, 78.0, 0.0, 0.0, 21.1, 0.268, 55.0],
[5.0, 106.0, 82.0, 30.0, 0.0, 39.5, 0.286, 38.0],
[2.0, 108.0, 52.0, 26.0, 63.0, 32.5, 0.318, 22.0],
[10.0, 108.0, 66.0, 0.0, 0.0, 32.4, 0.272, 42.0],
[4.0, 154.0, 62.0, 31.0, 284.0, 32.8, 0.237, 23.0],
[0.0, 102.0, 75.0, 23.0, 0.0, 0.0, 0.572, 21.0],
[9.0, 57.0, 80.0, 37.0, 0.0, 32.8, 0.096, 41.0],
[2.0, 106.0, 64.0, 35.0, 119.0, 30.5, 1.4, 34.0],
[5.0, 147.0, 78.0, 0.0, 0.0, 33.7, 0.218, 65.0],
[2.0, 90.0, 70.0, 17.0, 0.0, 27.3, 0.085, 22.0],
[1.0, 136.0, 74.0, 50.0, 204.0, 37.4, 0.399, 24.0],
[4.0, 114.0, 65.0, 0.0, 0.0, 21.9, 0.432, 37.0],
[9.0, 156.0, 86.0, 28.0, 155.0, 34.3, 1.189, 42.0],
[1.0, 153.0, 82.0, 42.0, 485.0, 40.6, 0.687, 23.0],
[8.0, 188.0, 78.0, 0.0, 0.0, 47.9, 0.137, 43.0],
[7.0, 152.0, 88.0, 44.0, 0.0, 50.0, 0.337, 36.0],
[2.0, 99.0, 52.0, 15.0, 94.0, 24.6, 0.637, 21.0],
[1.0, 109.0, 56.0, 21.0, 135.0, 25.2, 0.833, 23.0],
[2.0, 88.0, 74.0, 19.0, 53.0, 29.0, 0.229, 22.0],
[17.0, 163.0, 72.0, 41.0, 114.0, 40.9, 0.817, 47.0],
[4.0, 151.0, 90.0, 38.0, 0.0, 29.7, 0.294, 36.0],
[7.0, 102.0, 74.0, 40.0, 105.0, 37.2, 0.204, 45.0],
[0.0, 114.0, 80.0, 34.0, 285.0, 44.2, 0.167, 27.0],
[2.0, 100.0, 64.0, 23.0, 0.0, 29.7, 0.368, 21.0],
[0.0, 131.0, 88.0, 0.0, 0.0, 31.6, 0.743, 32.0],
[6.0, 104.0, 74.0, 18.0, 156.0, 29.9, 0.722, 41.0],
[3.0, 148.0, 66.0, 25.0, 0.0, 32.5, 0.256, 22.0],
[4.0, 120.0, 68.0, 0.0, 0.0, 29.6, 0.709, 34.0],
[4.0, 110.0, 66.0, 0.0, 0.0, 31.9, 0.471, 29.0],
[3.0, 111.0, 90.0, 12.0, 78.0, 28.4, 0.495, 29.0],
[6.0, 102.0, 82.0, 0.0, 0.0, 30.8, 0.18, 36.0],
[6.0, 134.0, 70.0, 23.0, 130.0, 35.4, 0.542, 29.0],
[2.0, 87.0, 0.0, 23.0, 0.0, 28.9, 0.773, 25.0],
[1.0, 79.0, 60.0, 42.0, 48.0, 43.5, 0.678, 23.0],
[2.0, 75.0, 64.0, 24.0, 55.0, 29.7, 0.37, 33.0],
[8.0, 179.0, 72.0, 42.0, 130.0, 32.7, 0.719, 36.0],
[6.0, 85.0, 78.0, 0.0, 0.0, 31.2, 0.382, 42.0],
[0.0, 129.0, 110.0, 46.0, 130.0, 67.1, 0.319, 26.0],
[5.0, 143.0, 78.0, 0.0, 0.0, 45.0, 0.19, 47.0],
[5.0, 130.0, 82.0, 0.0, 0.0, 39.1, 0.956, 37.0],
[6.0, 87.0, 80.0, 0.0, 0.0, 23.2, 0.084, 32.0],
[0.0, 119.0, 64.0, 18.0, 92.0, 34.9, 0.725, 23.0],
[1.0, 0.0, 74.0, 20.0, 23.0, 27.7, 0.299, 21.0],
[5.0, 73.0, 60.0, 0.0, 0.0, 26.8, 0.268, 27.0],
[4.0, 141.0, 74.0, 0.0, 0.0, 27.6, 0.244, 40.0],
[7.0, 194.0, 68.0, 28.0, 0.0, 35.9, 0.745, 41.0],
[8.0, 181.0, 68.0, 36.0, 495.0, 30.1, 0.615, 60.0],
[1.0, 128.0, 98.0, 41.0, 58.0, 32.0, 1.321, 33.0],
[8.0, 109.0, 76.0, 39.0, 114.0, 27.9, 0.64, 31.0],
[5.0, 139.0, 80.0, 35.0, 160.0, 31.6, 0.361, 25.0],
[3.0, 111.0, 62.0, 0.0, 0.0, 22.6, 0.142, 21.0],
[9.0, 123.0, 70.0, 44.0, 94.0, 33.1, 0.374, 40.0],
[7.0, 159.0, 66.0, 0.0, 0.0, 30.4, 0.383, 36.0],
[11.0, 135.0, 0.0, 0.0, 0.0, 52.3, 0.578, 40.0],
[8.0, 85.0, 55.0, 20.0, 0.0, 24.4, 0.136, 42.0],
[5.0, 158.0, 84.0, 41.0, 210.0, 39.4, 0.395, 29.0],
[1.0, 105.0, 58.0, 0.0, 0.0, 24.3, 0.187, 21.0],
[3.0, 107.0, 62.0, 13.0, 48.0, 22.9, 0.678, 23.0],
[4.0, 109.0, 64.0, 44.0, 99.0, 34.8, 0.905, 26.0],
[4.0, 148.0, 60.0, 27.0, 318.0, 30.9, 0.15, 29.0],
[0.0, 113.0, 80.0, 16.0, 0.0, 31.0, 0.874, 21.0],
[1.0, 138.0, 82.0, 0.0, 0.0, 40.1, 0.236, 28.0],
[0.0, 108.0, 68.0, 20.0, 0.0, 27.3, 0.787, 32.0],
[2.0, 99.0, 70.0, 16.0, 44.0, 20.4, 0.235, 27.0],
[6.0, 103.0, 72.0, 32.0, 190.0, 37.7, 0.324, 55.0],
[5.0, 111.0, 72.0, 28.0, 0.0, 23.9, 0.407, 27.0],
[8.0, 196.0, 76.0, 29.0, 280.0, 37.5, 0.605, 57.0],
[5.0, 162.0, 104.0, 0.0, 0.0, 37.7, 0.151, 52.0],
[1.0, 96.0, 64.0, 27.0, 87.0, 33.2, 0.289, 21.0],
[7.0, 184.0, 84.0, 33.0, 0.0, 35.5, 0.355, 41.0],
[2.0, 81.0, 60.0, 22.0, 0.0, 27.7, 0.29, 25.0],
[0.0, 147.0, 85.0, 54.0, 0.0, 42.8, 0.375, 24.0],
[7.0, 179.0, 95.0, 31.0, 0.0, 34.2, 0.164, 60.0],
[0.0, 140.0, 65.0, 26.0, 130.0, 42.6, 0.431, 24.0],
[9.0, 112.0, 82.0, 32.0, 175.0, 34.2, 0.26, 36.0],
[12.0, 151.0, 70.0, 40.0, 271.0, 41.8, 0.742, 38.0],
[5.0, 109.0, 62.0, 41.0, 129.0, 35.8, 0.514, 25.0],
[6.0, 125.0, 68.0, 30.0, 120.0, 30.0, 0.464, 32.0],
[5.0, 85.0, 74.0, 22.0, 0.0, 29.0, 1.224, 32.0],
[5.0, 112.0, 66.0, 0.0, 0.0, 37.8, 0.261, 41.0],
[0.0, 177.0, 60.0, 29.0, 478.0, 34.6, 1.072, 21.0],
[2.0, 158.0, 90.0, 0.0, 0.0, 31.6, 0.805, 66.0],
[7.0, 119.0, 0.0, 0.0, 0.0, 25.2, 0.209, 37.0],
[7.0, 142.0, 60.0, 33.0, 190.0, 28.8, 0.687, 61.0],
[1.0, 100.0, 66.0, 15.0, 56.0, 23.6, 0.666, 26.0],
[1.0, 87.0, 78.0, 27.0, 32.0, 34.6, 0.101, 22.0],
[0.0, 101.0, 76.0, 0.0, 0.0, 35.7, 0.198, 26.0],
[3.0, 162.0, 52.0, 38.0, 0.0, 37.2, 0.652, 24.0],
[4.0, 197.0, 70.0, 39.0, 744.0, 36.7, 2.329, 31.0],
[0.0, 117.0, 80.0, 31.0, 53.0, 45.2, 0.089, 24.0],
[4.0, 142.0, 86.0, 0.0, 0.0, 44.0, 0.645, 22.0],
[6.0, 134.0, 80.0, 37.0, 370.0, 46.2, 0.238, 46.0],
[1.0, 79.0, 80.0, 25.0, 37.0, 25.4, 0.583, 22.0],
[4.0, 122.0, 68.0, 0.0, 0.0, 35.0, 0.394, 29.0],
[3.0, 74.0, 68.0, 28.0, 45.0, 29.7, 0.293, 23.0],
[4.0, 171.0, 72.0, 0.0, 0.0, 43.6, 0.479, 26.0],
[7.0, 181.0, 84.0, 21.0, 192.0, 35.9, 0.586, 51.0],
[0.0, 179.0, 90.0, 27.0, 0.0, 44.1, 0.686, 23.0],
[9.0, 164.0, 84.0, 21.0, 0.0, 30.8, 0.831, 32.0],
[0.0, 104.0, 76.0, 0.0, 0.0, 18.4, 0.582, 27.0],
[1.0, 91.0, 64.0, 24.0, 0.0, 29.2, 0.192, 21.0],
[4.0, 91.0, 70.0, 32.0, 88.0, 33.1, 0.446, 22.0],
[3.0, 139.0, 54.0, 0.0, 0.0, 25.6, 0.402, 22.0],
[6.0, 119.0, 50.0, 22.0, 176.0, 27.1, 1.318, 33.0],
[2.0, 146.0, 76.0, 35.0, 194.0, 38.2, 0.329, 29.0],
[9.0, 184.0, 85.0, 15.0, 0.0, 30.0, 1.213, 49.0],
[10.0, 122.0, 68.0, 0.0, 0.0, 31.2, 0.258, 41.0],
[0.0, 165.0, 90.0, 33.0, 680.0, 52.3, 0.427, 23.0],
[9.0, 124.0, 70.0, 33.0, 402.0, 35.4, 0.282, 34.0],
[1.0, 111.0, 86.0, 19.0, 0.0, 30.1, 0.143, 23.0],
[9.0, 106.0, 52.0, 0.0, 0.0, 31.2, 0.38, 42.0],
[2.0, 129.0, 84.0, 0.0, 0.0, 28.0, 0.284, 27.0],
[2.0, 90.0, 80.0, 14.0, 55.0, 24.4, 0.249, 24.0],
[0.0, 86.0, 68.0, 32.0, 0.0, 35.8, 0.238, 25.0],
[12.0, 92.0, 62.0, 7.0, 258.0, 27.6, 0.926, 44.0],
[1.0, 113.0, 64.0, 35.0, 0.0, 33.6, 0.543, 21.0],
[3.0, 111.0, 56.0, 39.0, 0.0, 30.1, 0.557, 30.0],
[2.0, 114.0, 68.0, 22.0, 0.0, 28.7, 0.092, 25.0],
[1.0, 193.0, 50.0, 16.0, 375.0, 25.9, 0.655, 24.0],
[11.0, 155.0, 76.0, 28.0, 150.0, 33.3, 1.353, 51.0],
[3.0, 191.0, 68.0, 15.0, 130.0, 30.9, 0.299, 34.0],
[3.0, 141.0, 0.0, 0.0, 0.0, 30.0, 0.761, 27.0],
[4.0, 95.0, 70.0, 32.0, 0.0, 32.1, 0.612, 24.0],
[3.0, 142.0, 80.0, 15.0, 0.0, 32.4, 0.2, 63.0],
[4.0, 123.0, 62.0, 0.0, 0.0, 32.0, 0.226, 35.0],
[5.0, 96.0, 74.0, 18.0, 67.0, 33.6, 0.997, 43.0],
[0.0, 138.0, 0.0, 0.0, 0.0, 36.3, 0.933, 25.0],
[2.0, 128.0, 64.0, 42.0, 0.0, 40.0, 1.101, 24.0],
[0.0, 102.0, 52.0, 0.0, 0.0, 25.1, 0.078, 21.0],
[2.0, 146.0, 0.0, 0.0, 0.0, 27.5, 0.24, 28.0],
[10.0, 101.0, 86.0, 37.0, 0.0, 45.6, 1.136, 38.0],
[2.0, 108.0, 62.0, 32.0, 56.0, 25.2, 0.128, 21.0],
[3.0, 122.0, 78.0, 0.0, 0.0, 23.0, 0.254, 40.0],
[1.0, 71.0, 78.0, 50.0, 45.0, 33.2, 0.422, 21.0],
[13.0, 106.0, 70.0, 0.0, 0.0, 34.2, 0.251, 52.0],
[2.0, 100.0, 70.0, 52.0, 57.0, 40.5, 0.677, 25.0],
[7.0, 106.0, 60.0, 24.0, 0.0, 26.5, 0.296, 29.0],
[0.0, 104.0, 64.0, 23.0, 116.0, 27.8, 0.454, 23.0],
[5.0, 114.0, 74.0, 0.0, 0.0, 24.9, 0.744, 57.0],
[2.0, 108.0, 62.0, 10.0, 278.0, 25.3, 0.881, 22.0],
[0.0, 146.0, 70.0, 0.0, 0.0, 37.9, 0.334, 28.0],
[10.0, 129.0, 76.0, 28.0, 122.0, 35.9, 0.28, 39.0],
[7.0, 133.0, 88.0, 15.0, 155.0, 32.4, 0.262, 37.0],
[7.0, 161.0, 86.0, 0.0, 0.0, 30.4, 0.165, 47.0],
[2.0, 108.0, 80.0, 0.0, 0.0, 27.0, 0.259, 52.0],
[7.0, 136.0, 74.0, 26.0, 135.0, 26.0, 0.647, 51.0],
[5.0, 155.0, 84.0, 44.0, 545.0, 38.7, 0.619, 34.0],
[1.0, 119.0, 86.0, 39.0, 220.0, 45.6, 0.808, 29.0],
[4.0, 96.0, 56.0, 17.0, 49.0, 20.8, 0.34, 26.0],
[5.0, 108.0, 72.0, 43.0, 75.0, 36.1, 0.263, 33.0],
[0.0, 78.0, 88.0, 29.0, 40.0, 36.9, 0.434, 21.0],
[0.0, 107.0, 62.0, 30.0, 74.0, 36.6, 0.757, 25.0],
[2.0, 128.0, 78.0, 37.0, 182.0, 43.3, 1.224, 31.0],
[1.0, 128.0, 48.0, 45.0, 194.0, 40.5, 0.613, 24.0]]

#	var y = [[1],[0],[1],[0],[1],[0],[1],[0],[1],[0],
#			[1],[0],[1],[0],[1],[0],[1],[0],[1],[0],
#			[1],[0],[1],[0],[1],[0],[1],[0],[1],[0],
#			[1],[0],[1],[0],[1],[0],[1],[0],[1],[0]]
	var y = [[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0],
	[1],[0]]
	
	
	print("Dataset X size: ", len(X))
	print("Dataset X size: ", len(X[0]))
	print("Dataset y size: ", len(y))
	var input_size = 8
	var hidden_size = 1
	var output_size = 1
	
	# Initialize the weights
	var result = initialize_weights(input_size, hidden_size, output_size)
	var weights_input_hidden = result[0]
	var weights_hidden_output = result[1]
	
	# Train the neural network
	var epochs = 10
	var learning_rate = 0.1

	var start_time = Time.get_ticks_msec()
	for epoch in range(epochs):
		var result_ff = feedforward(X, weights_input_hidden, weights_hidden_output)
		var hidden_output = result_ff[0]
		var output_ff = result_ff[1]
		
		var result_bp = optim_sgd(X, y, learning_rate, weights_input_hidden, weights_hidden_output, hidden_output, output_ff)
		weights_input_hidden = result_bp[0]
		weights_hidden_output = result_bp[1]
	
	var end_time = Time.get_ticks_msec()
	var elapsed_time = end_time - start_time
	print("Time taken: " + str(elapsed_time/1000) + " s (" + str(elapsed_time) + " ms)")
	

	# Test the trained network
	var test_data = [[[6,148,72,35,0,33.6,0.627,50]],
	[[1,85,66,29,0,26.6,0.351,31]]]
	for data in test_data:
		var pred1 = feedforward(data, weights_input_hidden, weights_hidden_output)
		var prediction = pred1[1]
		print("Input: ", data)
		print("Output: ", prediction)