import numpy as np

from IPython import embed

# vis2ir
# mAP = {
#     1: 72.78,
#     2: 68.78,
#     3: 72.19,
#     4: 73.49,
#     5: 71.79,
#     6: 70.43,
#     7: 67.36,
#     8: 71.63,
#     9: 71.78,
#     10: 68.65,
# }

# r1 = {
#     1: 79.90,
#     2: 77.04,
#     3: 80.15,
#     4: 79.42,
#     5: 78.98,
#     6: 76.65,
#     7: 73.01,
#     8: 77.91,
#     9: 80.53,
#     10: 75.34,
# }

mAP = {
    1: 74.18,
    2: 72.61,
    3: 74.78,
    4: 76.64,
    5: 75.88,
    6: 74.15,
    7: 72.42,
    8: 77.17,
    9: 72.48,
    10: 72.69,
}

r1 = {
    1: 79.90,
    2: 79.22,
    3: 79.22,
    4: 80.97,
    5: 82.28,
    6: 80.78,
    7: 77.43,
    8: 81.36,
    9: 80.05,
    10: 78.54,
}

# ir2vis
mAP2 = {
    1: 71.33,
    2: 69.82,
    3: 68.85,
    4: 73.61,
    5: 69.17,
    6: 70.90,
    7: 69.61,
    8: 72.67,
    9: 71.40,
    10: 68.74,
}

r12 = {
    1: 77.14,
    2: 78.54,
    3: 75.63,
    4: 82.09,
    5: 77.77,
    6: 78.20,
    7: 76.02,
    8: 79.42,
    9: 80.39,
    10: 75.29,
}

print("r1: {}".format(np.mean(np.array(list(r1.values())))))
print("mAP: {}".format(np.mean(np.array(list(mAP.values())))))
print("r12: {}".format(np.mean(np.array(list(r12.values())))))
print("mAP2: {}".format(np.mean(np.array(list(mAP2.values())))))
