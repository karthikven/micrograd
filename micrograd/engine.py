import math

class Value:
    
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._children = set(_children)
        self._op = _op
        self.label = label 
    
    def __repr__(self):
        return f"Value(Data={self.data})"
    
    def __add__(self, other):
        if isinstance(other, int) or isinstance(other, float): other = Value(other)
        return Value(self.data + other.data, _children=(self, other), _op='+')
    
    def __mul__(self, other):
        if isinstance(other, int) or isinstance(other, float): other = Value(other)
        return Value(self.data * other.data, _children=(self, other), _op='*')

    def __pow__(self, other):
        assert isinstance(other, (float, int))
        out = Value(self.data ** other, _children=(self, ), _op= f"** {other}")
        return out

    def __radd__(self, other):
        return self + other
    
    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        return self * other**-1

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + -other
    
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        return Value(data=t, _children=(self, ), _op='tanh')
    
    def exp(self):
        out = math.exp(self.data)
        return Value(data=out, _children=(self, ), _op='exp')
        
    def update_grad_for_children(self):
        if self._op == '+':
            for c in self._children:
                c.grad += self.grad
        elif self._op == "*":
            c1, c2 = self._children
            c1.grad += self.grad * c2.data
            c2.grad += self.grad * c1.data
        elif self._op == "tanh":
            for c in self._children:
                c.grad += ((1 - (self.data)**2) * self.grad)
        elif self._op == "exp":
            for c in self._children:
                c.grad += (self.data * self.grad)
        elif len(self._op)>=2 and self._op[:2] == "**":
            # find the power self.data was raised to
            val = None
            for c in self._children:
                val = c.data
            exponent = math.log(self.data, val)
            for c in self._children:
                diff = exponent * (c.data ** (exponent - 1)) * self.grad
                c.grad += diff
        return
        
    def topo_sort(self):
        explored = set()
        topo_order = []
        def dfs(node):
            if node in explored:
                return
            if not node._children:
                topo_order.append(node)
                return 
            explored.add(node)
            for ch in node._children:
                if ch not in explored:
                    dfs(ch)
            topo_order.append(node)
            return 
        dfs(self)
        return reversed(topo_order)
        
    def backward(self):        
        topo_order = self.topo_sort()
        self.grad = 1.0
        for node in topo_order:
            node.update_grad_for_children()