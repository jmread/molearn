# MOLearn

Methods for Multiple-Output Learning in python

## About

For a maturing Java-based framework for multi-label multi-output learning, see the [MEKA](http://meka.sourceforge.net/) framework. But sometimes, it's nice to work in python, hence this project. The basic problem transformation methods are implemented, as in MEKA, except using [scikit-learn](http://scikit-learn.org/stable/) for base classifiers (rather than WEKA). 

I have also come across the [scikit-multilearn](http://scikit-multilearn.github.io/) with similar goals which also in fact has a wrapper to MEKA classifiers.

## Installation

```
	$ python setup.py install
```

If you install locally, then use the `--prefix` option, e.g., 

```
	$ python setup.py develop --prefix=$HOME/.local/
```

If you will be developing, then

```
	$ git clone https://github.com/jmread/cerebro
	$ cd cerebro
	$ python setup.py develop
```

## Running
 	
```
	$ python runDemo.py
```
