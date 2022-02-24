__all__ = ['LearningRateWarmup']


class LearningRateWarmup(object):
    def __init__(self, optimizer, warmup_iteration, target_lr, after_scheduler=None):
        self.optimizer = optimizer
        self.warmup_iteration = warmup_iteration
        self.target_lr = target_lr
        self.after_scheduler = after_scheduler
        self.global_step = 0
        self.warmup_lr = 0
        
    def warmup_learning_rate(self, curr_interation):
        self.warmup_lr = self.target_lr * curr_interation / self.warmup_iteration
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.warmup_lr
            
    def step(self):
        self.global_step += 1
        if self.global_step <= self.warmup_iteration:
            self.warmup_learning_rate(self.global_step)
        else:
            self.after_scheduler.step(self.global_step - self.warmup_iteration)
            
    def get_last_lr(self):
        if self.global_step <= self.warmup_iteration:
            return (self.warmup_lr,)
        else:
            return self.after_scheduler.get_last_lr()
            