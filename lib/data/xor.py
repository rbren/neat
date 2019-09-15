from random import randint

SET_SIZE = 1000

MAIN_DATA = [
  {'x': [0.0, 0.0], 'y': [0.0]},
  {'x': [1.0, 1.0], 'y': [0.0]},
  {'x': [1.0, 0.0], 'y': [1.0]},
  {'x': [0.0, 1.0], 'y': [1.0]},
]

def feed_dict(x, y_, train):
    xs = []
    ys = []
    for i in range(SET_SIZE):
        item = MAIN_DATA[randint(0, 3)]
        xs.append(item['x'])
        ys.append(item['y'])
    return {x: xs, y_: ys}

