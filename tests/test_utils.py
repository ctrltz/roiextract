import pytest

from roiextract.utils import _check_input, _report_props


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
    "param,value,allowed_values", [
        ["param1", 4, [1, 2, 3]],
        ["param2", "hom", ["sim", "rat"]]
    ]
)
def test_check_input_should_not_allow(param, value, allowed_values):
    with pytest.raises(ValueError):
        _check_input(param, value, allowed_values)


@pytest.mark.parametrize(
    "props,expected_result", [
        [dict(rat=0.812, sim=0.0626), "rat=0.81, sim=0.063"],
        [dict(rat=0.812, hom=0.2), "rat=0.81, hom=0.2"],
        [dict(rat=0.812, sim=0.0626, hom=0.2), "rat=0.81, sim=0.063, hom=0.2"],
    ]
)
def test_report_props(props, expected_result):
    assert _report_props(props) == expected_result