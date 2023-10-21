extends Node

func _ready():
	var file = FileAccess.open("/Users/danielmilne/Documents/GitHub/GodotMLFramework/diabetes.csv", FileAccess.READ)
	var X = PackedFloat32Array()
	var y = PackedFloat32Array()

	if file:
		while not file.eof_reached():
			var csv_line = file.get_csv_line(",")
			var x_row = PackedFloat32Array()
			for i in range(8):  # Get the first 8 columns
				x_row.append(float(csv_line[i]))
			X.append(x_row)
			y.append(float([float(csv_line[8])]))  # Get the 9th column and make it a one-element array
		file.close()
	else:
		print("File not found")

	
	# Testing vectors
#	var a1 = [2, 2]
#	var a2 = [2, 2]
#	var v1 = Vector2(2, 2)
#	var v2 = Vector2(a2[0], a2[1])
#	var v3 = v1.dot(v2)
#	print(a1)
#	print(v1)
#	print(v2)
#	print(v3)
#
#	var variable = PackedVector2Array([Vector2(1, 2), Vector2(3, 4)])
#	print(variable)
