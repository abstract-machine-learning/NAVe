# -*- coding: utf-8 -*-
# =============================================================================
# File: raf.py
# Updated: 02/05/2023
# =============================================================================
'''Define the Reduced Affine Form (RAF) abstract domain'''
# =============================================================================
# Dependencies:
#   ./abstract_domain.py
#   ../base.py
# =============================================================================

from __future__ import annotations
from typing import Type

import numbers
import numpy as np

from robustness.base import Vector
from robustness.utils.hyperplane import Hyperplane

from .abstract_domain import AbstractDomain
from robustness import Boolean, Integer, Number, String

class Raf(AbstractDomain):
    '''Represents the RAF abstract domain'''

    def __init__(self,
        center: Number = 0.0,
        linear = None,
        noise: Number = 0.0,
        dimensions: int = 0
    ) -> None:
        if linear is None:
            linear = np.zeros(dimensions)
        self.center = center
        self.linear = linear
        self.noise = abs(noise)
    
    def size(self):
        return self.linear.shape[0]
    
    def lowerbound(self):
        return self.center - np.sum(np.absolute(self.linear)) - self.noise
    
    def upperbound(self):
        return self.center + np.sum(np.absolute(self.linear)) + self.noise

    def is_number(self):
        return np.count_nonzero(self.linear) == 0 and self.noise == 0.0

    def is_single_variable(self):
        has_noise = 1 if self.noise != 0.0 else 0
        return np.count_nonzero(self.linear) + has_noise <= 1
    
    @staticmethod
    def intersect(
        interval1: Type[Raf] | None,
        interval2: Type[Raf] | None
    ) -> Type[Raf] | None:
        return None
    
    @staticmethod
    def join(
        interval1: Type[Raf] | None,
        interval2: Type[Raf] | None
    ) -> Type[Raf] | None:
        return None

    def dominates(self,
        other: Type[Raf] | Number
    ) -> Boolean:
        if issubclass(type(other), Raf):
            return (self - other).lowerbound() >= 0.0
        return self.lowerbound() >= other

    def dominated_by(self,
        other: Type[Raf] | Number
    ) -> Boolean:
        if issubclass(type(other), Raf):
            return (self - other).upperbound() <= 0.0
        return self.ub <= other

    def strictly_dominates(self,
        other: Type[Raf] | Number
    ) -> Boolean:
        if issubclass(type(other), Raf):
            return (self - other).lowerbound() > 0.0
        return self.lowerbound() > other
    
    def strictly_dominated_by(self,
        other: Type[Raf] | Number
    ) -> Boolean:
        if issubclass(type(other), Raf):
            return (self - other).upperbound() < 0.0
        return self.ub < other
    
    def to_string(self) -> String:
        return f"<{self.center}, {self.linear}, {self.noise}>: [{self.lowerbound()}, {self.upperbound()}]"

    def to_python_type(self) -> Vector[Number]:
        return [self.center, self.linear, self.noise]

    def square(self):
        noise = self.noise**2 + 2 * self.center * self.noise
        for i in range(1, self.size()):
            noise += self.linear[i]**2
            noise += abs(2 * self.linear[i] * self.noise)
            for j in range(i + 1, self.size()):
                noise += abs(2 * self.linear[i] * self.linear[j])
        return Raf(self.center**2, 2 * self.center * self.linear, noise)

    def __lt__(self,
        other: Type[Raf] | Number
    ) -> Boolean:
        if issubclass(type(other), Raf):
            return self.lowerbound() < other.upperbound() or (self.lowerbound() == other.lowerbound() and self.upperbound() < other.upperbound())
        return self.lb < other
    
    def __neg__(self):
        return Raf(-self.center, -self.linear, abs(self.noise))

    def __add__(self,
        other: Type[Raf] | Number
    ) -> Type[Raf]:
        if issubclass(type(other), Raf):
            return Raf(
                self.center + other.center,
                np.add(self.linear, other.linear),
                abs(self.noise) + abs(other.noise)
            )
        return Raf(self.center + other, self.linear, self.noise)

    def __sub__(self,
        other: Type[Raf] | Number
    ) -> Type[Raf]:
        if issubclass(type(other), Raf):
            return Raf(
                self.center - other.center,
                np.subtract(self.linear, other.linear),
                abs(self.noise) + abs(other.noise)
            )
        return Raf(self.center - other, self.linear, self.noise)
    
    def __mul__(self,
        other: Type[Raf] | Number
    ) -> Type[Raf]:
        if isinstance(other, numbers.Number):
            return Raf(
                other * self.center,
                other * self.linear,
                abs(other * self.noise)
            )
        linear = np.empty(self.linear.shape)
        x_norm_one = 0.0
        y_norm_one = 0.0
        xy = 0.0
        xy_abs = 0.0
        for i in range(0, self.size()):
            xy += self.linear[i] * other.linear[i]
            xy_abs += abs(self.linear[i] * other.linear[i])
            x_norm_one += abs(self.linear[i])
            y_norm_one += abs(other.linear[i])
            linear[i] = other.center * self.linear[i] + self.center * other.linear[i]
        return Raf(
            self.center * other.center + 0.5 * xy,
            linear,
            abs(other.center) * self.noise
            + abs(self.center) * other.noise
            + (x_norm_one + self.noise) * (y_norm_one + other.noise)
            - 0.5 * xy_abs,
        )
    
    def __abs__(self) -> Type[Raf]:
        if self.lowerbound() >= 0.0:
            return self
        elif self.upperbound() < 0.0:
            return -self
        elif self.is_single_variable() and True:
            c = self.center
            index = 0
            a = self.linear[index]
            for i in range(0, len(self.linear)):
                if self.linear[i] != 0:
                    index = i
                    a = self.linear[i]
                    break
            m = 0.5 * (abs(c + a) - abs(c - a))
            q = (c * (abs(c + a) - abs(c - a)) + a * (abs(c + a) + abs(c - a))) / (4 * a)
            epsilon = (-c * (abs(c + a) - abs(c - a)) + a * (abs(c + a) + abs(c - a))) / (4 * a)
            raf = Raf(q, np.zeros(len(self.linear)), epsilon)
            raf.linear[index] = m
            return raf
        n = self.size() + 2

        # RAF as hyperplane
        p = Hyperplane(dimensions=n)
        p.constant = self.center
        for i in range(0, self.size()):
            p.coefficients[i] = self.linear[i]
        p.coefficients[self.size()] = self.noise
        p.coefficients[self.size() + 1] = -1
        #print(p)

        # Finds n points
        points = np.zeros((n, n))

        # Finds minimum and maximum
        for i in range(0, self.size()):
            points[0][i] = +1 if self.linear[i] < 0.0 else -1
            points[1][i] = -1 if self.linear[i] < 0.0 else +1
        points[0][self.size()] = -1
        points[1][self.size()] = +1
        points[0][self.size() + 1] = abs(self.lowerbound())
        points[1][self.size() + 1] = abs(self.upperbound())

        # Finds other points
        for i in range(0, self.size()):
            points[i + 2][i] = 1
            points[i + 2][self.size() + 1] = abs(p(points[i + 2]))
        #print(points)

        # Finds hyperplane for those points (upperbound)
        X = np.matrix(points)
        k = np.ones((n, 1))
        A = np.matrix.dot(np.linalg.inv(X), k)
        h_top = Hyperplane(np.squeeze(np.asarray(A)), -1.0)
        #print(h_top)

        # find one intersection between raf (as hyperplane) and x_d+2 = 0
        # init X = 0 of size n - 1 (avoid x_d+2)
        # init nabla = gradient of hp (avoid x_d+2)
        # if x * nabla + k < 0:
        #  X += nabla / n
        # else
        #  X -n nabla / n
        # todo: make me sound :)
        x = np.zeros(n)
        nabla = np.array([p.coefficients[i] for i in range(0, n)])
        nabla[n - 1] = 0
        for i in range(1, 100):
            #print(f"    {np.dot(x, nabla) + p.constant}")
            if np.dot(x, nabla) == -p.constant:
                break
            elif np.dot(x, nabla) > -p.constant:
                x -= nabla / i
            else:
                x += nabla / i
        #print(x)

        # find parallel hyperplane passing through intersection: this is the lowebound
        h_bottom = Hyperplane(h_top.coefficients, h_top.constant - h_top(x))
        #print(h_bottom)

        # find middle hyperplane
        # todo: fix the messup
        h = Hyperplane(h_top.coefficients, 0.5 * (h_top.constant + h_bottom.constant))
        delta = abs(0.5 * (h_top.constant - h_bottom.constant) / -h.coefficients[n-1])
        #print("H: ", h)
        h_c = h / -h.coefficients[n - 1]
        #print("H corrected: ", h_c)
        #print("delta: ", delta)

        return Raf(h_c.constant, h_c.coefficients[0:n-2], abs(h_c.coefficients[n - 2]) + delta)
    
    def __pow__(self,
        n: Integer
    ) -> Type[Raf]:
        if n == 2:
            return self.square()
        result = self
        for i in range(1, n):
            result = result * self
        return result