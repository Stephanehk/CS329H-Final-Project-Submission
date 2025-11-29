import numpy as np
#for traffic env
# "Crash": 0.0,
#   "MeanVelocity": 1751.4,
#   "MinVehicleSpeed": 630.5,
#   "SquaredSpeedAboveMax": 0.0,
#   "FuelConsumption": 1750.6,
#   "Brake": 138.4,
#   "Headway": -102.0

pairs = []
prefs = []

#-----------preferences I personally would have--------------------

#prefer any trajectories where both mean and min speed are higher
# for _ in range(100):
#     a1 = np.random.uniform(10,5000)
#     b1 = np.random.uniform(0,a1)
#     c1 = np.random.uniform(0,b1)
#     d1 = np.random.uniform(0,5000)
#     e1 = np.random.uniform(0,500)
#     f1 = np.random.uniform(-500,0)

#     a2 = np.random.uniform(10,5000)
#     b2 = np.random.uniform(0,a2)
#     # c2 = np.random.uniform(0,b2)
#     d2 = np.random.uniform(d1-10,d1+10)
#     e2 = np.random.uniform(e1-10,e1+10)
#     f2 = np.random.uniform(f1-10,f1+10)
    
#     feat1 = [0, a1, b1, 0, d1, e1, f1]
#     feat2 = [0, a2, b2, 0, d2, e2, f2]

#     if a1 > a2*1.5 and b1 > b2*1.5:
#         pairs.append((feat1,feat2))
#         prefs.append(1)
#     elif a2 > a1*1.5 and b2 > b1*1.5:
#         pairs.append((feat2,feat1))
#         prefs.append(-1)

    
# #prefer trajectory that is slower but where the min speed is way higher than trajectoires where the mean speed if very high but the min speed is very low
# for _ in range(50):
#     a1 = np.random.uniform(10,5000)
#     c1 = np.random.uniform(0,b1)
#     d1 = np.random.uniform(0,5000)
#     e1 = np.random.uniform(0,500)
#     f1 = np.random.uniform(-500,0)

#     a2 = np.random.uniform(10,5000)
#     b2 = np.random.uniform(0,a2)
#     c2 = np.random.uniform(0,b2)
#     b1 = np.random.uniform(0,int(b2/5))

#     feat1 = [0, a1, b1, 0, 0, 0, 0]
#     feat2 = [0, a2, b2, 0, 0, 0, 0]

#     if b1*5 < b2 and a1 > a2:

#         pairs.append((feat1,feat2))
#         prefs.append(-1)


#preferences an agressive driver would have

#preferences a cautious driver would have


#----Below are preferences I am pretty sure any g.t. reward function should satisfy----
#always prefer no crash
for _ in range(50):
    a1 = np.random.uniform(10,5000)
    b1 = np.random.uniform(0,a1)
    c1 = np.random.uniform(0,b1)
    d1 = np.random.uniform(0,5000)
    e1 = np.random.uniform(0,500)
    f1 = np.random.uniform(-500,0)

    a2 = np.random.uniform(10,5000)
    b2 = np.random.uniform(0,a2)
    c2 = np.random.uniform(0,b2)
    d2 = np.random.uniform(0,5000)
    e2 = np.random.uniform(0,500)
    f2 = np.random.uniform(-500,0)
    
    feat1 = [1, a1, b1, c1, d1, e1, f1]
    feat2 = [0, a2, b2, c2, d2, e2, f2]
    pairs.append((feat1,feat2))
    prefs.append(-1)

#prefer higher mean velocity all else equal
for _ in range(50):
    a = np.random.uniform(0,5000)
    b = np.random.uniform(0,a)
    c = np.random.uniform(0,b)
    d = np.random.uniform(0,5000)
    e = np.random.uniform(0,500)
    f = np.random.uniform(-500,0)
    
    feat1 = [0, a, b, c, d, e, f]
    feat2 = [0, a-np.random.uniform(10,a), b, c, d, e, f]
    pairs.append((feat1,feat2))
    prefs.append(1)

#prefer higher min vehicle speed all else equal
for _ in range(50):
    a = np.random.uniform(20,5000)
    b = np.random.uniform(10,a)
    c = np.random.uniform(0,b)
    d = np.random.uniform(0,5000)
    e = np.random.uniform(0,500)
    f = np.random.uniform(-500,0)
    
    feat1 = [0, a, b, c, d, e, f]
    feat2 = [0, a, b-np.random.uniform(10,b), c, d, e, f]
    pairs.append((feat1,feat2))
    prefs.append(1)

#prefer lower squared speed above max all else equal
for _ in range(50):
    a = np.random.uniform(20,5000)
    b = np.random.uniform(10,a)
    c = np.random.uniform(0,b)
    d = np.random.uniform(0,5000)
    e = np.random.uniform(0,500)
    f = np.random.uniform(-500,0)

    feat1 = [0, a, b, c, d, e, f]
    feat2 = [0, a, b, c+np.random.uniform(10,500), d, e, f]
    pairs.append((feat1,feat2))
    prefs.append(1)

#prefer lower fuel consumption all else equal
for _ in range(50):
    a = np.random.uniform(20,5000)
    b = np.random.uniform(10,a)
    c = np.random.uniform(0,b)
    d = np.random.uniform(0,5000)
    e = np.random.uniform(0,500)
    f = np.random.uniform(-500,0)
    
    f1 = [0, a, b, c, d, e, f]
    f2 = [0, a, b, c, d+np.random.uniform(10,5000), e, f]
    pairs.append((f1,f2))
    prefs.append(1)

#prefer lower brake all else equal
for _ in range(50):
    a = np.random.uniform(20,5000)
    b = np.random.uniform(10,a)
    c = np.random.uniform(0,b)
    d = np.random.uniform(0,5000)
    e = np.random.uniform(0,500)
    f = np.random.uniform(-500,0)
    
    f1 = [0, a, b, c, d, e, f]
    f2 = [0, a, b, c, d, e+np.random.uniform(10,500), f]
    pairs.append((f1,f2))
    prefs.append(1)

#prefer higher headway all else equal
for _ in range(50):
    a = np.random.uniform(20,5000)
    b = np.random.uniform(10,a)
    c = np.random.uniform(0,b)
    d = np.random.uniform(0,5000)
    e = np.random.uniform(0,500)
    f = np.random.uniform(-500,0)
    
    f1 = [0, a, b, c, d, e, f]
    f2 = [0, a, b, c, d, e, min(f+np.random.uniform(10,500), 0)]
    pairs.append((f1,f2))
    prefs.append(-1)

np.save("data/stephane_traffic_hand_designed_init_traj_pairs.npy", pairs)
np.save("data/stephane_traffic_hand_designed_init_traj_prefs.npy", prefs)