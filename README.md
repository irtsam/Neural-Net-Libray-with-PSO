This project is based on the UCI Devanagri Characte Recognition Databases. The project is implemented in Python and only NumPy i.e. no machine learning framework is used. The project implements:

Utilities.py: Functions to help define a linear Neural Net
Train.py: Training of linear Neural Net using Gradient descent with momentum
Dataset Loader: Dataset Loader to read images and create labels. In addition also defines function to get dataset once created.
Train.py (Test): Testing of the neural net is done using a test function in train.py.
The following libraries are needed:

Numpy
OpenCV
This is the link for the dataset: https://archive.ics.uci.edu/ml/datasets/Devanagari+Handwritten+Character+Dataset

Download the dataset and change the path argument in the path variable in train.py.

Run the train.py.

Thats it!

#TODO: Add PSO optimizer file to Optimizers #TODO: Change learning part to function/class call one optimizers defined in Optimizers file.