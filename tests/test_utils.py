import pytest

from roiextract.utils import _check_input


@pytest.mark.parametrize(
    "param,value,allowed_values", 
    [
        ["param1", 2, [1, 2, 3]],
        ["param2", "sim", ["sim", "rat"]]
    ]
)
def test_check_input_should_allow(param, value, allowed_values):
    _check_input(param, value, allowed_values)


@pytest.mark.parametrize(
    "param,value,allowed_values", 
    [
        ["param1", 4, [1, 2, 3]],
        ["param2", "hom", ["sim", "rat"]]
    ]
)
def test_check_input_should_not_allow(param, value, allowed_values):
    with pytest.raises(ValueError):
        _check_input(param, value, allowed_values)