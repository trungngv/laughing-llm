from program import *
def test_lowercase_conversion():
    assert to_lowercase("HELLO") == "hello"

def test_lowercase_conversion_with_numbers():
    assert to_lowercase("123ABC") == "123abc"

def test_lowercase_conversion_with_special_characters():
    assert to_lowercase("!@#ABC$%^") == "!@#abc$%^"

def test_lowercase_conversion_with_spaces():
    assert to_lowercase("HELLO WORLD") == "hello world"

def test_lowercase_conversion_with_empty_string():
    assert to_lowercase("") == ""

def test_lowercase_conversion_with_already_lowercase():
    assert to_lowercase("hello") == "hello"

