import numpy as np

def main():
    T = np.matrix("1; 2; 3; 4; 5")
    Y = np.matrix("0.8; 2.1; 3; 4.1; 5")
    Yh = lambda p: p[0] * T + p[1]
    P_init = np.array([2, -1])

    print(Y)
    print(Yh(P_init))
    print(jacobian(Yh, P_init))

    opt = Optimizer(Y, Yh, P_init)
    opt.set_step(0.01)
    opt.set_iteration(10000)
    P = opt.optimize()
    print(P)

class Optimizer:
    def __init__(self, Y, Yh, P):
        self.Y = Y   # Vector
        self.Yh = Yh # Function
        self.P = P   # Vector

    # Setter
    def set_step(self, alpha):
        self.alpha = alpha

    def set_iteration(self, num):
        self.num = num

    # Getter
    def get_sse(self):
        error = (self.Y - self.Yh(self.P))
        return (error.T * error)[0,0]

    def get_param(self):
        return self.P

    def update(self):
        J = jacobian(self.Yh, self.P)
        h = 2 * self.alpha * J * (self.Y - self.Yh(self.P))
        h = np.squeeze(np.array(np.squeeze(h)))
        self.P = self.P + h

    def optimize(self):
        for i in range(self.num):
            self.update()
            print(self.get_sse())
        return self.get_param()

def jacobian(Yh, P):
    m = len(P)
    n = len(Yh(P))
    h = np.zeros(m)
    J = np.zeros((m, n))
    for i in range(m):
        h_i = h
        h_i[i] = 1e-6
        P_i = P + h_i
        Yh_i = Yh(P_i) - Yh(P)
        for j in range(n):
            J[i, j] = Yh_i[j] / 1e-6
    return J

if __name__ == "__main__":
    main()
