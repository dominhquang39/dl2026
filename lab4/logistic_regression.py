import math

def sigmoid(z):
    return 1.0 / (1 + math.exp(-z))

def loss_function(N, x1, x2, y, w): # Cross entropy
    total_loss = 0
    for i in range(N):
        y_hat = w[0] + w[1] * x1[i][0] + w[2] * x2[i][0]
        yi = y[i][0]
        total_loss += yi * y_hat - math.log(1 + math.exp(y_hat))
    return - total_loss / N

def gradient(N, x1, x2, y, w):
    grad = [0.0, 0.0, 0.0]
    for i in range(N):
        x1_i = x1[i][0]
        x2_i = x2[i][0]
        yi = y[i][0]
        y_hat = w[0] + w[1] * x1_i + w[2] * x2_i
        grad[0] += 1 - yi - sigmoid(-y_hat)
        grad[1] += -yi*x1_i + x1_i * (1 - sigmoid(-y_hat))
        grad[2] += -yi*x2_i + x2_i * (1 - sigmoid(-y_hat))

    grad[0] = grad[0] / N
    grad[1] = grad[1] / N
    grad[2] = grad[2] / N
    return grad

def gradient_descent(x1, x2, y, lr, threshold=1e-6):
    N = len(y)
    w = [0.0, 1.0, 2.0]
    count = 0
    converged = False
    loss = float('inf')

    while not converged:
        grad = gradient(N, x1, x2, y, w)
        w[0] -= lr*grad[0]
        w[1] -= lr*grad[1]
        w[2] -= lr*grad[2]
        new_loss = loss_function(N, x1, x2, y, w)
        count += 1
        if count % 10000 == 0:
            print(f"Iteration {count}: Loss={new_loss:.3f}")

        if abs(new_loss - loss) < threshold:
            converged = True
            print(f"Model converged after {count} iterations")
        loss = new_loss

    print('Final weights:')
    print(f'w0 = {w[0]:.3f}, w1 = {w[1]:.3f}, w2 = {w[2]:.3f}')
    return w