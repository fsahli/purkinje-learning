# PurkinjeECG

This repository extends `fractal-tree` with

- [x] Eikonal solution along the tree
- [x] One-way coupling to Propeiko
- [x] One-way coupling to `fim-python`
- [x] Bi-directional coupling (antidromic activation)
- [x] ECG computation

Propeiko for colab is provided as binary in `bin/propeiko`.

## Paraview time

Add a Programmable Filter, below the Script and RequestInformation

```python
def GetUpdateTimestep(algorithm):
      """Returns the requested time value, or None if not present"""
      executive = algorithm.GetExecutive()
      outInfo = executive.GetOutputInformation(0)
      if not outInfo.Has(executive.UPDATE_TIME_STEP()):
          return None
      return outInfo.Get(executive.UPDATE_TIME_STEP())
 
req_time = GetUpdateTimestep(self)

#output = self.GetOutput()
#input = self.GetInput()

#pdata = inputs[0].PointData["u_current"] * req_time
#output.PointData.append(pdata,"ufrac")

import numpy as np

V0,V1 = -80,20 # mV
eps = 1.0 # ms

input0=inputs[0]
u=input0.PointData["u_current"]
umin,umax = np.min(u),np.max(u)
uu = u-umin
Vm = V0 + (V1-V0)/2*(1+np.tanh( (req_time - uu)/eps ))
output.PointData.append(Vm, "Vm")

output.GetInformation().Set(output.DATA_TIME_STEP(), req_time)
```

```python
def SetOutputTimesteps(algorithm, timesteps):
      executive = algorithm.GetExecutive()
      outInfo = executive.GetOutputInformation(0)
      outInfo.Remove(executive.TIME_STEPS())
      for timestep in timesteps:
          outInfo.Append(executive.TIME_STEPS(), timestep)
      outInfo.Remove(executive.TIME_RANGE())
      outInfo.Append(executive.TIME_RANGE(), timesteps[0])
      outInfo.Append(executive.TIME_RANGE(), timesteps[-1])

import numpy as np
ts = np.linspace(0,200,201)
SetOutputTimesteps(self, ts)
```