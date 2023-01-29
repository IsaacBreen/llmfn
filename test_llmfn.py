from llmfn import Arguments
from llmfn import FunctionExample
from llmfn import llmfn
from llmfn import make_llmfn


def test_make_blackbox():
    blackbox = make_llmfn(
        examples=[
            FunctionExample(arguments=Arguments.call(2, 3), output=8),
            FunctionExample(arguments=Arguments.call(4, 2), output=16),
            FunctionExample(arguments=Arguments.call(3, 4), output=81),
        ],
        function_name="f",
        decoder=eval,
    )
    assert blackbox(2, 3) == 8
    assert blackbox(3, 3) == 27
    assert blackbox(3, 4) == 81
    assert blackbox(5, 3) == 125


def test_approximate():
    @llmfn(
        examples=[
            FunctionExample(arguments=Arguments.call(2, 3), output=8),
            FunctionExample(arguments=Arguments.call(4, 2), output=16),
            FunctionExample(arguments=Arguments.call(3, 4), output=81),
        ],
        decoder=eval,
    )
    def f(x: int, y: int) -> int:
        return x ** y

    assert f(2, 3) == 8
    assert f(3, 3) == 27
    assert f(3, 4) == 81
    assert f(5, 3) == 125
