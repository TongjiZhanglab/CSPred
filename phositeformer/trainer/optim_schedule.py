import numpy as np

class ScheduledOptim(object):
    "A simple wrapper class for learning rate scheduling."

    def __init__(self, optimizer, init_lr, n_warmup_steps):
        """
        :parameter n_warmup_steps: Due to the random initialization of weights 
                                   at the beginning of training, choosing a larger 
                                   learning rate can lead to model instability (oscillation). 
                                   By using the Warmup learning rate strategy, we can start 
                                   training with a smaller learning rate for a few epochs or 
                                   steps. With the lower learning rate during the warmup phase, 
                                   the model can gradually stabilize. Once the model is relatively 
                                   stable, we can then switch to the predetermined learning rate 
                                   for training, which helps accelerate convergence and achieve 
                                   better model performance.
        """
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = init_lr # np.power(d_model, -0.5)

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

    def _update_learning_rate(self):
        "Learning rate scheduling per step."

        self.n_current_steps += 1
        
        if self.n_current_steps < self.n_warmup_steps:
            warmup_percent_done = self.n_current_steps / self.n_warmup_steps
            # gradual warmup_lr
            lr = self.init_lr * warmup_percent_done
    
        else:
            # 预热学习率结束后,学习率呈指数衰减(近似模拟指数衰减)
            lr = self.init_lr ** 1.0001 ** (self.n_current_steps - self.n_warmup_steps)
                    
        
        # lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr
            
            