_base_ = './tct_detection.py'
# dataset
part = 'normal'
data = dict(
    train = dict(part = part),
    val = dict(part = part),
    test = dict(part = part))
