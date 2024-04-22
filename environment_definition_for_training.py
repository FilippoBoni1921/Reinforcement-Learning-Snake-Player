import environment_fully_observable 
import environment_partially_observable


def get_env(n,type="full",mask_size=2):
    # n is the number of boards that you want to simulate parallely
    # size is the size of each board, also considering the borders
    # mask for the partially observable, is the size of the local neighborhood
    size = 7

    if type == "full":
        e = environment_fully_observable.OriginalSnakeEnvironment(n, size)
    
    elif type == "partial":
        e = environment_partially_observable.OriginalSnakeEnvironment(n, size, mask_size)

    return e
