import torch.optim as optim

class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, learning_rate, minimum_learning_rate, epoch_decay_rate, 
            grad_clip=2, n_warmup_steps=0, summarywriter=None):
        self._optimizer = optimizer
        self.n_current_steps = 0
        self.lr = learning_rate
        self.mlr = minimum_learning_rate
        self.decay = epoch_decay_rate
        self.grad_clip = grad_clip
        self.n_warmup_steps = n_warmup_steps
        self.summarywriter = summarywriter

    def clip_gradient(self):
        for group in self._optimizer.param_groups:
            for param in group['params']:
                param.grad.data.clamp_(-self.grad_clip, self.grad_clip)

    def step(self):
        "Step with the inner optimizer"
        self.step_update_learning_rate()
        #self.clip_gradient()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def epoch_update_learning_rate(self):
        if self.n_current_steps > self.n_warmup_steps:
            self.lr = max(self.mlr, self.decay * self.lr)

    def step_update_learning_rate(self):
        ''' Learning rate scheduling per step '''
        self.n_current_steps += 1
        ratio = min(self.n_current_steps / (self.n_warmup_steps + 1.0), 1)
        learning_rate = self.lr * ratio

        if self.summarywriter is not None:
            self.summarywriter.add_scalar('learning_rate', learning_rate, global_step=self.n_current_steps)

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = learning_rate

    def get_lr(self):
        return self.lr

def get_optimizer(opt, model, summarywriter=None):
    optim_mapping = {
        'adam': optim.Adam,
        'rmsprop': optim.RMSprop,
    }

    optim_type = opt['optim'].lower()
    assert optim_type in optim_mapping.keys()

    return ScheduledOptim(
        optimizer=optim_mapping[optim_type](
            filter(lambda p: p.requires_grad, model.parameters()), weight_decay=opt["weight_decay"]),
        learning_rate=opt['learning_rate'],
        minimum_learning_rate=opt['minimum_learning_rate'],
        epoch_decay_rate=opt['decay'],
        n_warmup_steps=opt.get('n_warmup_steps', 0),
        summarywriter=summarywriter
        )
