from b2_data.pipeline import announce as announce_pipeline
from b3_network.network import announce as announce_network
from b4_loss.loss import announce as announce_loss
from b5_backprop.backprop import announce as announce_backprop
from b6_optimizer.optimizer import announce as announce_optimizer
from b7_training.trainer import announce as announce_trainer

announce_pipeline()
announce_network()
announce_loss()
announce_backprop()
announce_optimizer()
announce_trainer()

