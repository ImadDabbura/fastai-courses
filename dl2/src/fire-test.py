'''
An example script to illustrate the use of `Fire` package. It helps in
generating CLI from any Python object such as class or function or even the 
whole script itself.
'''
import fire


def hello(name='World'):
    '''Test funtion.'''
    return f'Hello {name}!'


if __name__ == '__main__':
    fire.Fire(hello)
