from abc import ABC, abstractmethod

class Scheduler(ABC):
    def __init__(self, e_0, e_f, n_steps) -> None:
        self.e_0 = e_0
        self.e_f = e_f
        self.n_steps = n_steps
    @abstractmethod
    def step(self, t):
        pass
    
class LinearScheduler(Scheduler):
    def __init__(self, e_0, e_f, n_steps) -> None:
        super().__init__(e_0, e_f, n_steps)
        self.slope = (e_0 - e_f)/n_steps
        
    def step(self, t):
        e_t = self.slope*t + self.e_0
        return max(e_t, self.e_f)
       
class ExpDecayScheduler(Scheduler):
    def __init__(self, e_0, e_f, n_steps) -> None:
        super().__init__(e_0, e_f, n_steps)
        self.gamma = (e_f / e_0)**(1/n_steps)
        
    def step(self, t):
        e_t = self.e_0 * self.gamma**t
        return max(e_t, self.e_f)
  
class PowerDecayScheduler(Scheduler):
    def __init__(self, e_0, e_f, n_steps) -> None:
        super().__init__(e_0, e_f, n_steps)
        def gauss_sum(a:int, n:int, inc:float):
            return n/2*(2*a+(n-1)*inc)
        
        k = gauss_sum(a=0, n=n_steps, inc=1.0)
        self.gamma = (e_f / e_0)**(1/k)
        self.acc = e_0
        
    def step(self, t):
        self.acc *= self.gamma**t
        return max(self.acc, self.e_f)
    
        