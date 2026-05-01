import json
import queue as queue_module
import threading
import contextlib
import io

from flask import Flask, jsonify, request, Response
from flask_cors import CORS
from b2_data.pipeline import load_data, get_batch
from b3_network.network import Network
from b4_loss.loss import compute_loss
from b5_backprop.backprop import backward
from b6_optimizer.optimizer import step

app = Flask(__name__)
CORS(app, origins=['http://localhost:3000', 'http://localhost:5173'])

_event_queue: queue_module.Queue = queue_module.Queue()
_trained_net = None


def _train_worker(config):
    global _trained_net
    epochs = config['epochs']
    with contextlib.redirect_stdout(io.StringIO()):
        data = load_data()
        net = Network(config['layer_sizes'])
    for epoch in range(epochs):
        with contextlib.redirect_stdout(io.StringIO()):
            batch = get_batch(data, config['batch_size'])
            preds = net.forward(batch['X'])
            loss, grad = compute_loss(preds, batch['y'])
            grads = backward(net, grad)
            step(net, grads, config['lr'])
        _event_queue.put({'epoch': epoch + 1, 'total': epochs, 'loss': round(float(loss), 4)})
    _trained_net = net
    _event_queue.put({'status': 'done'})


@app.get('/api/data')
def get_data():
    data = load_data()
    X = data['X']
    return jsonify({
        'shape': list(X.shape),
        'rows': X[:3].tolist(),
    })


@app.get('/api/config')
def get_config():
    return jsonify({'layer_sizes': [4, 8, 1]})


@app.post('/api/train')
def post_train():
    config = request.get_json() or {}
    config.setdefault('epochs', 3)
    config.setdefault('batch_size', 32)
    config.setdefault('lr', 0.01)
    config.setdefault('layer_sizes', [4, 8, 1])
    threading.Thread(target=_train_worker, args=(config,), daemon=True).start()
    return jsonify({'status': 'started'})


@app.get('/api/events')
def get_events():
    def stream():
        while True:
            try:
                event = _event_queue.get(timeout=60)
                yield f"data: {json.dumps(event)}\n\n"
                if event.get('status') == 'done':
                    break
            except queue_module.Empty:
                yield ": keepalive\n\n"
    return Response(
        stream(),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
    )


@app.get('/api/predict')
def get_predict():
    if _trained_net is None:
        return jsonify({'error': 'No trained model available'}), 400
    index = int(request.args.get('index', 0))
    with contextlib.redirect_stdout(io.StringIO()):
        data = load_data()
    x = data['X'][index].reshape(1, 4)
    output = _trained_net.forward(x)
    return jsonify({
        'input':  [round(float(v), 4) for v in data['X'][index]],
        'output': [round(float(v), 4) for v in output.flatten()],
        'label':  round(float(data['y'][index]), 4),
    })


@app.get('/api/status')
def get_status():
    return jsonify({'status': 'idle'})


if __name__ == '__main__':
    app.run(port=5001, debug=True)
