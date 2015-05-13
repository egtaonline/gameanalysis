_CONFIG = {
    'big_tests': False
}


def config(**kwargs):
    '''Configure which tests are run

    big_tests - run large tests which may fail if you lack memory or cpu

    '''
    _CONFIG.update(kwargs)
