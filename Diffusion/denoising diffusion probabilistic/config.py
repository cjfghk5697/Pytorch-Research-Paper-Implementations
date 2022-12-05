"""
All of params is based on DDPM paper.
 T(Int) : 1000
 beta(Float) : 1e-4
 end(Float) : 0.02
"""
params={
    'image_size':64,
    'batch_size':64,
    'pin_memory':False,
    'num_workers':4,
    'epochs':300,
    'lr':1e-3,
    'T':200,
    'beta':0.0001,
    'end':0.02
}