import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

#"""
def log_base(x, base):
    return np.log(x) / np.log(base)

end = 1e6
inc = 1
t = np.arange(0, end+inc, inc)
n = len(t)
a = t[0]
e_0 = 0.9
e_f = 0.1

gamma = (e_f/e_0)**(1/end)
y = e_0*gamma**t
plt.plot(t, y, label='Exp decay: f(x)')

y = ((e_f - e_0)/end) * t + e_0
plt.plot(t, y, label='Linear decay: f(x)')

########################################################################################
k = n/2*(2*a+(n-1)*inc)
l = 1

gamma = (e_f/e_0)**(1/(k*l))
print("Gamma:", gamma)
y = []
acc = e_0
for i in t:
    acc = acc * gamma**(i*l)
    y.append(acc)

y = np.array(y)
plt.plot(t, y, label=f'Pro. Exp decay: f(x) | lambda: {l}')

plt.legend()
plt.savefig('./example_decays.png')
