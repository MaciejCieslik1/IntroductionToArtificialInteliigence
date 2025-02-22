from typing import Callable, Tuple
import matplotlib.pyplot as plt
import numpy as np


class SimpleGradientDescent:
    X = np.arange(-2, 2, 0.05)
    Y = np.arange(-3, 2, 0.05)
    X, Y = np.meshgrid(X, Y)

    def __init__(
        self,
        func: Callable[[float, float], float],
        grad_func: Callable[[float, float], Tuple[float, float]],
        alpha: float = 0.1,
    ):
        self.alpha = alpha
        self.func = func
        self.grad_func = grad_func
        self.trace = np.empty((0, 2))  # trace of search

    def _calc_Z_value(self):
        self.Z = self.func(self.X, self.Y)

    def plot_func(self, file_name):
        self._calc_Z_value()
        plt.figure()
        plt.contour(self.X, self.Y, self.Z, 50)
        if len(self.trace) > 0:
            plt.scatter(self.trace[:, 0], self.trace[:, 1], s=10, color='red')
            plt.title('Gradient Descent Visualization')
            plt.xlabel('X1')
            plt.ylabel('X2')
            plt.xlim(-2, 2)
            plt.xticks(np.arange(-2, 2.5, 0.5))
            plt.ylim(-1.5, 1.5) 
            plt.yticks(np.arange(-3, 2.5, 0.5))
            plt.savefig(file_name)
            plt.show()

    def calculate_func_vale(self, x1: float, x2: float) -> float:
        return self.func(x1, x2)

    def calculate_func_grad(self, x1: float, x2: float) -> Tuple[float, float]:
        return self.grad_func(x1, x2)

    def gradient_descent_step(self, x1: float, x2: float) -> Tuple[float, float]:
        x1 = x1 - self.alpha * self.calculate_func_grad(x1, x2)[0]
        x2 = x2 - self.alpha * self.calculate_func_grad(x1, x2)[1]
        return x1, x2

    def minimize(
        self,
        x1_init: float,
        x2_init: float,
        steps: int,
        verbose: int = 0,
        plot: bool = False,
    ) -> float:
        x1 = x1_init
        x2 = x2_init
        for step in range(steps):
            self.trace = np.vstack((self.trace, [x1, x2]))
            x1, x2 = self.gradient_descent_step(x1, x2)
            value = self.calculate_func_vale(x1, x2)
            if verbose:
                print(f"Current step: {step}, x1 value: {x1}, x2 value: {x2}, func value: {value}")
        if plot:
            self.plot_func(f"plot_n_{self.alpha}_x1_{x1_init}_x2_{x2_init}_func_{self.func.__name__[-2:]}.png")
        return value


def calculate_fx(x1: float, x2: float) -> float:
    return x1 ** 2 + x2 ** 2


def calculate_gx(x1: float, x2: float) -> float:
    # 1.5 - e^(-x1^2 -x2^2) - 0.5 * e^[-(x1-1)^2 -(x2 + 2)^2]
    return 1.5 - np.exp(-x1 ** 2 - x2 ** 2) - 0.5 * np.exp(-(x1-1) ** 2 - (x2 + 2) ** 2)


def calculate_fx_gradient(x1: float, x2: float) -> Tuple[float, float]:
    return 2 * x1, 2 * x2


def calculate_gx_gradient(x1: float, x2: float) -> Tuple[float, float]:
    # 2*x1*e^(-x1^2 -x2^2) - (x1 - 1) * e^[-(x1-1)^2 -(x2 + 2)^2]
    # 2*x2*e^(-x1^2 -x2^2) - (x2 + 2) * e^[-(x1-1)^2 -(x2 + 2)^2]
    value_for_x1 = 2 * x1 * np.exp(-x1 ** 2 - x2 ** 2) - (x1 - 1) * np.exp(-(x1 - 1) ** 2 - (x2 + 2) ** 2)
    value_for_x2 = 2 * x2 * np.exp(-x1 ** 2 - x2 ** 2) - (x2 + 2) * np.exp(-(x1 - 1) ** 2 - (x2 + 2) ** 2)
    return value_for_x1, value_for_x2


solution1 = SimpleGradientDescent(calculate_fx, calculate_fx_gradient)
solution1.minimize(1, 1, 20, True, True)

# solution2 = SimpleGradientDescent(calculate_gx, calculate_gx_gradient)
# solution2.minimize(1, 1, 20, True, True)

# solution3 = SimpleGradientDescent(calculate_fx, calculate_fx_gradient, 0.3)
# solution3.minimize(-1.5, 0.5, 20, True, True)

# solution4 = SimpleGradientDescent(calculate_gx, calculate_gx_gradient, 0.3)
# solution4.minimize(-1.5, 0.5, 20, True, True)

# solution5 = SimpleGradientDescent(calculate_fx, calculate_fx_gradient, 0.5)
# solution5.minimize(2, 1.1, 20, True, True)

# solution6 = SimpleGradientDescent(calculate_gx, calculate_gx_gradient, 0.5)
# solution6.minimize(2, 1.1, 20, True, True)

# solution7 = SimpleGradientDescent(calculate_fx, calculate_fx_gradient, 0.7)
# solution7.minimize(0, 1.5, 20, True, True)

# solution8 = SimpleGradientDescent(calculate_gx, calculate_gx_gradient, 0.7)
# solution8.minimize(0, 1.5, 20, True, True)

# solution9 = SimpleGradientDescent(calculate_fx, calculate_fx_gradient, 0.9)
# solution9.minimize(2, -0.5, 20, True, True)

# solution10 = SimpleGradientDescent(calculate_gx, calculate_gx_gradient, 0.9)
# solution10.minimize(2, -0.5, 20, True, True)

# solution11 = SimpleGradientDescent(calculate_fx, calculate_fx_gradient, 1)
# solution11.minimize(-1, 3, 20, True, True)

# solution12 = SimpleGradientDescent(calculate_gx, calculate_gx_gradient, 1)
# solution12.minimize(-1, 3, 20, True, True)

# solution13 = SimpleGradientDescent(calculate_fx, calculate_fx_gradient, 1.5)
# solution13.minimize(0.71, 1.7, 20, True, True)

# solution14 = SimpleGradientDescent(calculate_gx, calculate_gx_gradient, 1.5)
# solution14.minimize(0.71, 1.7, 20, True, True)

# solution15 = SimpleGradientDescent(calculate_fx, calculate_fx_gradient, 2)
# solution15.minimize(-1.3, -1.3, 20, True, True)

# solution16 = SimpleGradientDescent(calculate_gx, calculate_gx_gradient, 2)
# solution16.minimize(-1.3, -1.3, 20, True, True)

# solution17 = SimpleGradientDescent(calculate_fx, calculate_fx_gradient, 0.05)
# solution17.minimize(0.5, -0.75, 20, True, True)

# solution18 = SimpleGradientDescent(calculate_gx, calculate_gx_gradient, 0.05)
# solution18.minimize(0.5, -0.75, 20, True, True)

# solution19 = SimpleGradientDescent(calculate_fx, calculate_fx_gradient, 0.01)
# solution19.minimize(0.35, -0.15, 20, True, True)

# solution20 = SimpleGradientDescent(calculate_gx, calculate_gx_gradient, 0.01)
# solution20.minimize(0.35, -0.15, 20, True, True)
