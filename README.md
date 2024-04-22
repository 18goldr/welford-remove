# Welford-Remove
This library is a Python (Numpy) implementation of a modified Welford's algorithm,
which is online and parallel algorithm for calculating variances. Typically, Welford's algorithm
only allows for adding data points. This modification allows for removing data points.

Welford's algorithm is described in the following:

* [Wikipedia:Welford Online Algorithm](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm)
* [Wikipedia:Welford Parallel Algorithm](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm)

The modification for removing data points is described here:
* [StackOverflow Post](https://stackoverflow.com/questions/30876298/removing-a-prior-sample-while-using-welfords-method-for-computing-single-pass-v)

Welford's original method is more numerically stable than the standard method as
described in the following blog:
* [Accurately computing running variance](www.johndcook.com/blog/standard_deviation)
    
However, There has been no formal analysis on whether
the modified version of the algorithm provided here is numerically stable, but based 
on the testing done in test_welford.test_remove, I have reason to believe it is.

This library is inspired by the jvf's implementation, which is implemented
without using numpy library. In particular, this implementation is a fork
of the implementation by a-mitani,
    * Implementation done by jvf: github.com/jvf/welford
    * Implementation done by a-mitani: github.com/a-mitani/welford

## Install
Download package via [PyPI repository](https://pypi.org/project/welford-with-remove/)
```
$ pip install welford
```

## Example
### For Online Calculation
```python
import numpy as np
from welford import Welford

# Initialize Welford object
w = Welford()

# Input data samples sequentially
w.add(np.array([0, 100]))
w.add(np.array([1, 110]))
w.add(np.array([2, 120]))

# output
print(w.mean)  # mean --> [1. 110.]
print(w.var_s)  # sample variance --> [1. 100.]
print(w.var_p)  # population variance --> [0.6666. 66.66.]

# You can add other samples after calculating variances.
w.add(np.array([3, 130]))
w.add(np.array([4, 140]))

# output with added samples
print(w.mean)  # mean --> [2. 120.]
print(w.var_s)  # sample variance --> [2.5. 250.]
print(w.var_p)  # population variance --> [2. 200.]

# You can remove samples after calculating variances.
w.remove(np.array([3, 130]))
w.remove(np.array([4, 140]))
print(w.mean)  # mean --> [1. 110.]
print(w.var_s)  # sample variance --> [1. 100.]
print(w.var_p)  # population variance --> [0.6666. 66.66.]

# You can also get the standard deviation
print(w.std_s)  # sample standard deviation --> [1. 10.]
print(w.std_p)  # population standard deviation --> [0.81649658. 8.16496581.]
```

Welford object supports initialization with data samples and batch addition of samples.
```python
import numpy as np
from welford import Welford

# Initialize Welford object with samples.
ini = np.array([[0, 100], [1, 110], [2, 120]])
w = Welford(ini)

# output
print(w.mean)  # mean --> [1. 110.]
print(w.var_s)  # sample variance --> [1. 100.]
print(w.var_p)  # population variance --> [0.66666667. 66.66666667.]

# add other samples through batch method
other_samples = np.array([[3, 130], [4, 140]])
w.add_all(other_samples)

# output with added samples
print(w.mean)  # mean --> [2. 120.]
print(w.var_s)  # sample variance --> [2.5 250.]
print(w.var_p)  # population variance --> [2. 200.]
```

### For Parallel Calculation
Welford also offers parallel calculation method for variance.
```python
import numpy as np
from welford import Welford

# Initialize two Welford objects
w_1 = Welford()
w_2 = Welford()

# Each object will calculate variance of each samples in parallel.
# On w_1
w_1.add(np.array([0, 100]))
w_1.add(np.array([1, 110]))
w_1.add(np.array([2, 120]))
print(w_1.var_s)  # sample variance --> [1. 100.]
print(w_1.var_p)  # population variance --> [0.66666667. 66.66666667.]

# On w_2
w_2.add(np.array([3, 130]))
w_2.add(np.array([4, 140]))
print(w_2.var_s)  # sample variance --> [0.5 50.]
print(w_2.var_p)  # sample variance --> [0.25 25.]

# You can Merge objects to get variance of WHOLE samples
w_1.merge(w_2)
print(w.var_s)  # sample variance --> [2.5. 250.]
print(w_1.var_p)  # sample variance --> [2. 200.]
```