# simple_mnist.jl
a simple neural network on mnist dataset written in julia with no autodifferentiation

You must have recent version of julia installed.

```
git clone git@github.com:jaesunglee9/simple_mnist.jl.git
```


inside repo
in julia REPL:
```
]
activate .
{BACKSPACE}
include(initialize.jl)
```


if you wish to train the network, run:
```
gradient_descent!(network)
```
