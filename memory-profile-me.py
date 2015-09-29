import copy
import memory_profiler


@profile
def function():
    x = list(range(1000000))
    y = copy.deepcopy(x)
    del x
    return y

if __name__ == '__main__':
    function()
