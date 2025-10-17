# multilayer_perceptron
42 project - learning artificial neural networks and implement multilayer perceptron from scratch

Steps to do

1. Data preprocess and visualization
    - basic visualization with histogram, pairplot (done)
    - data cleaning: remove outlier, nan, empty, etc. (dataset is clean without missing data or outliers)
    - feature selection / extraction (consider to ignore feature 5, 9, 10 which did not help that much to distinguish M and B)
    - feature scaling (normalization, stadardization)
2. Train test split program
3. MLP training program
    - need to be able to config hidden layers with a config file
    - visualize learning process (loss, accuracy by epochs)
    - [bonus] store the history of metrics during training
    - [bonus] implement early stopping
    - [bonus] Use more complex optimization function lke Adam, RMSprop, ...
    - [bonus] Evaluate learning pahse with multiple metrics (not just a single R2 score for example)
4. prediction program
5. Display multiple learning curves on the same graph to compare different models (with different configs)


Ten real-valued features are computed for each cell nucleus:

	a) radius (mean of distances from center to points on the perimeter)
	b) texture (standard deviation of gray-scale values)
	c) perimeter
	d) area
	e) smoothness (local variation in radius lengths)
	f) compactness (perimeter^2 / area - 1.0)
	g) concavity (severity of concave portions of the contour)
	h) concave points (number of concave portions of the contour)
	i) symmetry 
	j) fractal dimension ("coastline approximation" - 1)

# References:

1. [MLP explained with visual](https://medium.com/data-science/multilayer-perceptron-explained-a-visual-guide-with-mini-2d-dataset-0ae8100c5d1c)
