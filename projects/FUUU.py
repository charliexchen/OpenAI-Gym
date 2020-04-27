
import random
def areSameEshan(a, b):
    for elem in a:
        try:
            b.remove(elem**2)
        except ValueError:
            return False
    return bool(b)


def areSameCharlie(a, b):
    set_b = {}
    for i in b:
        if i not in set_b:
            set_b[i] = 1
        else:
            set_b[i] += 1
    for i in a:
        k = i ** 2
        if k not in set_b:
            return False
        else:
            set_b[k] -= 1
            if set_b[k] == 0:
                set_b.pop(k)
    return bool(set_b)

def areSameCharlieSlow(a, b):
    hb = {}
    for i in b:
        if i not in hb:
            hb[i] = 1
        else:
            hb[i] += 1
    ha = {}
    for i in a:
        if i*i not in ha:
            ha[i*i] = 1
        else:
            ha[i*i] += 1
    return ha==hb


def testAreSame(f):
    #assert f([], []) == True
    #assert f([1, 2, 3], [1, 4, 9]) == True
    #assert not f([1, 2, 2], [1, 4, 9])
    #assert not f([1, 2, 3, 4], [1, 4, 9])
    #assert f([1, 2, 2, 3, 3, 3], [9,4,9,1,4,9]) == True

    massive_list = list(range(100000))*10
    massiver_sq_list = [i**2 for i in massive_list]+[1]
    random.shuffle(massiver_sq_list)
    import time
    first = time.time()
    f(massive_list, [i**2 for i in massive_list])
    end = time.time()
    print("time taken: ", end - first)


testAreSame(areSameCharlie)
testAreSame(areSameCharlieSlow)
testAreSame(areSameEshan)