def f(x):
    return x**2

def df(x):
    return 2*x

def gradient_descent(f, df, x0, lr, iter):
    x = x0

    print(f"Initial guess: {x}, learning rate: {lr}")

    for i in range(iter):
        fx = f(x)
        print(f"Iteration {i+1}: x={x:.3f}, f(x)={fx:.3f}")
        grad = df(x)
        x -= lr*grad

if __name__ == '__main__':
    learning_rate = [0.001, 0.005, 0.1, 0.5]
    for lr in learning_rate:
        gradient_descent(f, df, x0=5, lr=lr, iter=20)



