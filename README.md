# llmfn

**llmfn** is a Python library to approximate a function using OpenAI's API. You can use it to easily train a language model to approximate your own functions with few-shot prompting.

## Installation

You can install the package from pip:

```
pip install llmfn
```

## Usage

First, you need to set your OpenAI API key:

```python
import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")
```

Then, you can define a list of examples of the function's behavior:

```python
from llmfn import Arguments
from llmfn import FunctionExample

examples = [
    FunctionExample(arguments=Arguments.call(2, 3), output=5),
    FunctionExample(arguments=Arguments.call(5, 7), output=12),
    # ...
]
```

Finally, you can use the llmfn decorator to create an approximated version of your function:

```python
from llmfn import llmfn

@llmfn(examples=examples, function_name="my_function")
def my_function(a: int, b: int) -> int:
    return a + b

assert my_function(2, 3) == 5
```

Alternatively, you can use the `make_llmfn` function to create an approximated version of your function without using the decorator:

```python
from llmfn import make_llmfn

blackbox = make_llmfn(examples=examples, function_name="my_function")

assert blackbox(2, 3) == 5
```

## Advanced Usage

### Changing the Decoder

By default, the decoder is set to `lambda x: x`, which simply returns the output as a string. You can change the decoder to parse the output into a different data type:

```python
from llmfn import make_llmfn

def decoder(output: str) -> int:
    return int(output)

blackbox = make_llmfn(examples=examples, function_name="my_function", decoder=decoder)

assert blackbox(2, 3) == 5
```

The most useful decoder (and the most dangerous) is `eval`, which will evaluate the output as Python code:

```python
from llmfn import make_llmfn

blackbox = make_llmfn(examples=examples, function_name="my_function", decoder=eval)

assert blackbox(2, 3) == 5
```

Use this with caution - it could be used to execute arbitrary code. (This is why it's not the default decoder.)

### Changing the Engine

By default, the engine is set to `text-davinci-003`. You can change the engine to a different OpenAI engine:

```python
from llmfn import make_llmfn

blackbox = make_llmfn(examples=examples, function_name="my_function", engine="text-curie-001")

assert blackbox(2, 3) == 5
```

## Limitations

- The function's output must be a string.
- The approximated function can only handle arguments that are compatible with the examples.
- The approximated function may not work with complex or large functions.
- The API usage may be subject to rate limits and other restrictions imposed by OpenAI.

## Contributing

We welcome contributions to this project. If you have any ideas or suggestions, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.