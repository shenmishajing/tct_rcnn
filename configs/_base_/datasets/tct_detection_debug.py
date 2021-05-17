_base_ = './tct_detection.py'
data = dict(
    train = dict(debug_len = 4),
    val = dict(debug_len = 4),
    test = dict(debug_len = 4))
