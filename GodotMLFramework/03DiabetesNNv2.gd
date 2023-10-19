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
