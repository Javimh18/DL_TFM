import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

#"""
def log_base(x, base):
    return np.log(x) / np.log(base)

end = 10
inc = 1
t = np.arange(0, end+inc, inc)
n = len(t)
a = t[0]
e_0 = 0.9
e_f = 0.1
l = 1

gamma = (e_f/e_0)**(1/end)
y = e_0*gamma**t
plt.plot(t, y, label=f'Exp. decay with lambda={l} \n and gamma={gamma:.8f}')

y = ((e_f - e_0)/end) * t + e_0
plt.plot(t, y, label=f'Linear decay')

########################################################################################
k = n/2*(2*a+(n-1)*inc)

gamma = (e_f/e_0)**(1/(k*l))
y = []
acc = e_0
for i in t:
    acc = acc * gamma**(i*l)
    print(gamma**(i*l))
    y.append(acc)

y = np.array(y)
plt.plot(t, y, label=f'Product Exp. decay with lambda={l} \n and gamma={gamma:.12f}')

plt.legend()
plt.savefig('./example_decays.png')
