import pytest
import torch
from micrograd.engine import Value

def test_value_add():
    a = Value(2.0)
    b = Value(3.0)
    c = a + b
    assert c.data == 5.0

def test_value_add_integer():
    a = Value(4.0)
    b = 3 + a
    assert b.data == 7.0

def test_value_mul():
    a = Value(2.0)
    b = Value(3.0)
    c = a * b
    assert c.data == 6.0

def test_value_pow():
    a = Value(2.0)
    b = a ** -3
    assert b.data == 0.125

def test_value_div():
    a = Value(6.0)
    b = Value(2.0)
    c = a / b
    assert c.data == 3.0

def test_value_neg():
    a = Value(2.0)
    b = -a
    assert b.data == -2.0

def test_value_sub():
    a = Value(5.0)
    b = Value(3.0)
    c = a - b
    assert c.data == 2.0

def test_value_tanh():
    a = Value(0.0)
    b = a.tanh()
    assert abs(b.data) < 1e-6  # tanh(0) should be very close to 0

def test_value_exp():
    a = Value(1.0)
    b = a.exp()
    assert abs(b.data - 2.718281828) < 1e-6  # e^1 ≈ 2.718281828

def test_backward():
    # inputs
    x1 = Value(2.0, label="x1")
    x2 = Value(0.0, label="x2")
    # weights w1, w2
    w1 = Value(-3.0, label="w1")
    w2 = Value(1.0, label="w2")
    # bias
    b = Value(6.8813735, label="b")
    #x1w1 + x2w2 + b
    x1w1 = x1* w1; x1w1.label = "x1w1"
    x2w2 = x2* w2; x2w2.label = "x2w2"
    x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = "x1w1x2w2"
    n = x1w1x2w2 + b; n.label = "n"
    ###
    e = (2*n).exp()
    o = (e-1)/(e+1)
    ####
    o.label="o"
    o.backward()
    assert o.grad == 1.0 and round(x1.grad,1) == -1.5 and round(w1.grad, 1) == 1.0 and round(x2.grad, 1) == 0.5 and round(w2.grad, 1) == 0.0 and round(b.grad, 1) == 0.5

def test_backward_compare_torch():
    x1 = torch.Tensor([2.0]).double(); x1.requires_grad = True
    x2 = torch.Tensor([0.0]).double(); x2.requires_grad = True

    w1 = torch.Tensor([-3.0]).double(); w1.requires_grad = True
    w2 = torch.Tensor([1.0]).double(); w2.requires_grad = True

    b = torch.Tensor([6.8813735]).double(); b.requires_grad = True

    n = x1*w1 + x2*w2 + b
    o = torch.tanh(n)
    o.backward()

    x1_v = Value(2.0, label="x1")
    x2_v = Value(0.0, label="x2")
    w1_v = Value(-3.0, label="w1")
    w2_v = Value(1.0, label="w2")
    b_v = Value(6.8813735, label="b")

    n_v = x1_v*w1_v + x2_v*w2_v + b_v
    o_v = n_v.tanh()

    o_v.backward()
    assert round(w1.grad.item(), 1) == round(w1_v.grad, 1) and round(x1.grad.item(), 1) == round(x1_v.grad, 1) and round(w2.grad.item(), 1) == round(w2_v.grad, 1) and round(x2.grad.item(), 1) == round(x2_v.grad, 1) and round(b.grad.item(), 1) == round(b_v.grad, 1)

def test_backward_negative_power():
    # Create input value
    x = Value(2.0, label="x")
    
    # Define function: f(x) = x^(-2) + 3x
    f = x**(-2) + 3*x
    f.label = "f"
    
    # Compute backward pass
    f.backward()
    
    # Expected gradient: df/dx = -2x^(-3) + 3
    expected_grad = -2 * (2.0**(-3)) + 3
    
    # Check if computed gradient matches expected gradient
    assert abs(x.grad - expected_grad) < 1e-6, f"Expected grad: {expected_grad}, got: {x.grad}"
    
    # Additional test with a different value
    y = Value(0.5, label="y")
    g = y**(-3) + 2*y
    g.label = "g"
    g.backward()
    
    expected_grad_y = -3 * (0.5**(-4)) + 2
    assert abs(y.grad - expected_grad_y) < 1e-6, f"Expected grad: {expected_grad_y}, got: {y.grad}"

    