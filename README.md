# GodotMLFramework
An ML framework for Godot using only GDScript - no Python Godot wrappers here!

Folder is the Godot project. The .py of the same name of a .gd script is the equivalent.

Still working things out, but now have the ability to define, train and predict using a model (check out 03DiabetesNN).

Next steps:
- Improve speed
- More activation functions
- More initialization types
- More layer types
- Ability to load ONNX model

## 03DiabetesNN
In GDScript:
- Imports dataset as CSV
- Allows the user to define a fully connected network of arbituary size, like this:
```
network = [fully_connected(8, 8, 'sigmoid'),
		fully_connected(8, 8, 'sigmoid'),
		fully_connected(8, 1, 'sigmoid')]
```

Using the above network and 100 epochs:
- The Python version (03DiabetesNN.py) trains in 0.0286 s and predicts in 0.0002 s
- The GDSCript version (03DiabetesNN.gd) trains in 18 s and predicts in 0.017 s

Next aim is to reduce the GDScript training and inference times


## Functions
sigmoid<br>
sigmoid_derivative<br>
relu<br>
relu_derivative<br>
dot_product<br>
transpose<br>
square<br>
add<br>
subtract<br>
matmul<br>
mul_array_by_scalar<br>

train_network<br>
