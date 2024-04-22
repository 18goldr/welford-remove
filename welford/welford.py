#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This library is a Python (Numpy) implementation of a modified Welford's algorithm,
which is online and parallel algorithm for calculating variances. Typically, Welford's algorithm
only allows for adding data points. This modification allows for removing data points.

Welford's algorithm is described in the following:

* https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
* https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

The modification for removing data points is described here:
* https://stackoverflow.com/questions/30876298/removing-a-prior-sample-while-using-welfords-method-for-computing-single-pass-v

Welford's original method is more numerically stable than the standard method as
described in the following blog.
    * Accurately computing running variance: www.johndcook.com/blog/standard_deviation

However, There has been no formal analysis on whether
the modified version of the algorithm provided here is numerically stable, but based
on the testing done in test_welford.test_remove, I have reason to believe it is.

This library is inspired by the jvf's implementation, which is implemented
without using numpy library. In particular, this implementation is a fork
of the implementation by a-mitani,
    * Implementation done by jvf: github.com/jvf/welford
    * Implementation done by a-mitani: github.com/a-mitani/welford
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
from typing import Optional


class Welford:
    """Accumulator object for Welford's online and parallel variance algorithm which provides the ability to remove data points.

    Attributes:
        count (int): The number of accumulated samples.
        mean (np.array(D,)): Mean of the accumulated samples.
        var_s (np.array(D,)): Sample variance of the accumulated samples.
        var_p (np.array(D,)): Population variance of the accumulated samples.
        std_s (np.array(D,)): Sample standard deviation of the accumulated samples.
        std_p (np.array(D,)): Population standard deviation of the accumulated samples.
    """

    def __init__(self, elements: Optional[npt.NDArray[npt.NDArray]] = None):
        """
        Initialize with an optional data. 
        For calculation efficiency, Welford's method is not used in the initialization process.

        Args:
            elements: The data points to initialize with.

        """
        self.__count: int = 0
        self.__shape: Optional[tuple]
        self.__m: Optional[npt.NDArray]
        self.__s: Optional[npt.NDArray]
        self.__m_old: Optional[npt.NDArray]
        self.__s_old: Optional[npt.NDArray]
        self.__count_old: Optional[int]

        # Initialize instance attributes
        if elements is None:
            self.__shape = None
            # Current attribute values
            self.__count = 0
            self.__m = None
            self.__s = None
            # Previous attribute values for rolling back
            self.__count_old = None
            self.__m_old = None
            self.__s_old = None
        else:
            self.__shape = elements[0].shape
            # Current attribute values
            self.__count = elements.shape[0]
            self.__m = np.array(np.mean(elements, axis=0))
            self.__s = np.array(np.var(elements, axis=0, ddof=0) * elements.shape[0])
            # Previous attribute values for rolling back
            self.__count_old = None
            self.__init_old_with_nan()

    @property
    def count(self) -> Optional[int]:
        """Get the number of accumulated samples."""
        return self.__count

    @property
    def mean(self) -> Optional[npt.NDArray]:
        """Get the mean."""
        return self.__m

    @property
    def var_s(self) -> Optional[npt.NDArray]:
        """Get the sample variance."""
        return self.__get_var_with_ddof(ddof=1)

    @property
    def var_p(self) -> Optional[npt.NDArray]:
        """Get the population variance."""
        return self.__get_var_with_ddof(ddof=0)

    @property
    def std_s(self) -> Optional[npt.NDArray]:
        """Get the sample standard deviation."""
        if self.var_s is None:
            return None

        return np.sqrt(self.var_s, dtype=np.longdouble)

    @property
    def std_p(self) -> Optional[npt.NDArray]:
        """Get the population standard deviation."""
        if self.var_p is None:
            return None

        return np.sqrt(self.var_p, dtype=np.longdouble)

    def add(self, element: npt.NDArray, backup_flg: bool = True) -> None:
        """Add one data point.

        Args:
            element: The data point to add.
            backup_flg: If True, backup current state for rolling back.

        """

        # Initialize if not done so.
        if self.__shape is None:
            self.__shape = element.shape
            self.__m = np.zeros(element.shape)
            self.__s = np.zeros(element.shape)
            self.__init_old_with_nan()
            self.__count = 0
        else:
            # Argument check if already initialized
            assert element.shape == self.__shape

        if backup_flg:
            self.__backup_attrs()

        # Welford's algorithm
        self.__count += 1
        delta = element - self.__m
        self.__m += delta / self.__count
        self.__s += delta * (element - self.__m)

    def add_all(self, elements: npt.NDArray[npt.NDArray], backup_flg: bool = True) -> None:
        """Add multiple data points.

        Args:
            elements: The data points to add.
            backup_flg: If True, backup current state for rolling back.

        """
        if backup_flg:
            self.__backup_attrs()

        for elem in elements:
            self.add(elem, backup_flg=False)

    def remove(self, element: npt.NDArray, backup_flg: bool = True) -> None:
        """Remove one data point.

        Using a method derived from the original Welford's algorithm,
        we can remove a data point. Note however, there has been
        no analysis on whether this is numerically stable. See
        https://stackoverflow.com/questions/30876298/removing-a-prior-sample-while-using-welfords-method-for-computing-single-pass-v
        for more information.

        """

        if self.__shape is None:
            return
        else:
            assert element.shape == self.__shape

        if backup_flg:
            self.__backup_attrs()

        # The reverse of Welford's algorithm.
        old_m = np.copy(self.__m)
        self.__m -= (element - self.__m) / (self.count - 1)
        self.__s -= (element - self.__m) * (element - old_m)
        self.__count -= 1

    def remove_all(self, elements: npt.NDArray[npt.NDArray], backup_flg: bool = True) -> None:
        """Remove multiple data points.

        Args:
            elements: The data points to remove.
            backup_flg: If True, backup current state for rolling back.

        """
        if backup_flg:
            self.__backup_attrs()

        for elem in elements:
            self.remove(elem, backup_flg=False)

    def rollback(self) -> None:
        """
        Rollback to a prior state that has been saved.
        In order for a state to be saved, you must call
        pass backup_flag = True to either the method merge,
        add, remove, add_all, or remove_all."""

        if self.__shape is None:
            return

        self.__count = self.__count_old
        self.__m[...] = self.__m_old[...]
        self.__s[...] = self.__s_old[...]

    def merge(self, other: Welford, backup_flg: bool = True) -> None:
        """Merge this accumulator with another one.

        Args:
            other: The other accumulator to merge with.
            backup_flg: If True, backup current state for rolling back.

        """

        if other.__shape is None:
            pass
        elif self.__shape is None:
            if backup_flg:
                self.__backup_attrs()

            self.__count = other.__count
            self.__m = other.__m
            self.__s = other.__s
        else:
            if backup_flg:
                self.__backup_attrs()

            count = self.__count + other.__count
            delta = self.__m - other.__m
            delta2 = delta * delta
            m = (self.__count * self.__m + other.__count * other.__m) / count
            s = self.__s + other.__s + delta2 * (self.__count * other.__count) / count

            self.__count = count
            self.__m = m
            self.__s = s

    def __get_var_with_ddof(self, ddof: int) -> Optional[npt.NDArray]:
        """
        Get the variance with a given delta degrees of freedom (ddof).
        If there are fewer sample accumulated than the ddof, return a matrix
        filled with NaN values of the appropriate shape.

        Args:
            ddof: The delta degrees of freedom.

        """
        if self.__count <= 0:
            return None
        min_count = ddof
        if self.__count <= min_count:
            return np.full(self.__shape, np.nan)
        else:
            return self.__s / (self.__count - ddof)

    def __backup_attrs(self) -> None:
        """
        Backup current values for mean and variance. Used for when you want to rollback changes
        made through adding or removing samples.
        """
        if self.__shape is None:
            pass
        else:
            self.__count_old = self.__count
            self.__m_old[...] = self.__m[...]
            self.__s_old[...] = self.__s[...]

    def __init_old_with_nan(self) -> None:
        """
        Initialize old values with NaN.
        """
        self.__m_old = np.empty(self.__shape)
        self.__m_old[...] = np.nan
        self.__s_old = np.empty(self.__shape)
        self.__s_old[...] = np.nan
