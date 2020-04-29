# encoding: utf-8


import h5py

""" DATA """
x = 100
a = [1, 2, 3, 4, 5]

""" WRITE """
with h5py.File('output.h5', 'w') as f:
    f.create_dataset('x', data=x)
    f.create_dataset('a', data=a)

""" READ """
with h5py.File('output.h5', 'r') as f:
    print(f.keys())  # => ['a', 'x']
    print(f['x'].value)  # => 100
    print(f['a'].value)  # => [1 2 3 4 5]
    print(f['a'].shape)  # => (5,)
    print(f['a'].dtype)  # => int64

""" MAKE A GROUPED HDF """
with h5py.File('output_group.h5', 'w') as f:
    """ CODE 1 """
    f.create_group('hoge')
    f['hoge'].create_dataset('x', data=x)
    f['hoge'].create_dataset('a', data=a)

    """
    with the above procedure, we can get a following folder structure.
    
     output_group.h5 --- Group 'hoge'
                           |
                           +- Dataset 'x'
                           |
                           +- Dataset 'a'
    """

    """ CODE 2 """
    f.create_group('fuga')
    f.create_dataset('/fuga/x', data=200)
    f.create_dataset('/fuga/a', data=[10, 20, 30, 40, 50])
    # can skip to create_group
    # can remove '/' of '/fuga/x' ('fuga/x')

""" WRITE A GROUPED HDF """
with h5py.File('output_group.h5', 'r') as f:
    """ CODE 1 """
    print(f['hoge']['x'].value)  # => 100
    print(f['hoge']['a'].value)  # => [1 2 3 4 5]

    """ CODE 1 """
    print(f['fuga/x'].value)  # => 200
    print(f['fuga/a'].value)  # => [10 20 30 40 50]
    # can remove '/' of '/fuga/x' ('fuga/x')
