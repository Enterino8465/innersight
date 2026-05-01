import io
import contextlib

from b2_data.pipeline import load_data, get_batch
from b3_network.network import Network
from b4_loss.loss import compute_loss
from b5_backprop.backprop import backward
from b6_optimizer.optimizer import step


def train(config):
    with contextlib.redirect_stdout(io.StringIO()):
        data = load_data()
        net = Network(config['layer_sizes'])

    for epoch in range(config['epochs']):
        with contextlib.redirect_stdout(io.StringIO()):
            batch = get_batch(data, config['batch_size'])
            preds = net.forward(batch['X'])
            loss, grad = compute_loss(preds, batch['y'])
            grads = backward(net, grad)
            step(net, grads, config['lr'])
        print(f"Epoch {epoch+1}/{config['epochs']} | loss={loss:.4f}")

    print('Training complete.')
    return net
