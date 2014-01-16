ManifoldWarping
===============

Code for the
[AAAI 2012 Manifold Warping paper](http://people.cs.umass.edu/~ccarey/pubs/ManifoldWarping.pdf)

There are two independent implementations here, in Python and Matlab.

## Python Notes

Depends on [numpy](http://www.numpy.org/),
[scipy](http://www.scipy.org/),
[matplotlib](http://matplotlib.org/),
and [scikit-learn](http://scikit-learn.org/stable/).

Run `python test.py` to make sure everything works as intended,
then `python demo.py` to see (most of) the aligners in action.

## Matlab Notes

All dependencies are included.
To get started, run the `setup.m` script,
which will attempt to compile any MEX functions and adds support code to the matlab path.

To run the example code, try `manifold_warping_test.m` and `test_semisupervised.m`.

All of the alignment methods are implemented in the `Alignment` directory.
