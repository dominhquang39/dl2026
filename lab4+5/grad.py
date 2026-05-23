def gradient_descent(compute_grad, compute_loss, epochs=1000, threshold=1e-6):
    prev_loss = None
    for epoch in range(1, epochs + 1):
        compute_grad()
        if epoch % 1000 == 0:
            loss = compute_loss()
            print(f"Epoch {epoch} | Loss: {loss:.6f}")
            if prev_loss is not None and abs(prev_loss - loss) < threshold:
                print(f"Model converged at epoch {epoch}.")
                break
            prev_loss = loss