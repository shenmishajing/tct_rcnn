_base_ = './tct_detection.py'
references = 3
data = dict(
    train = dict(references = references),
    val = dict(references = references),
    test = dict(references = references))
