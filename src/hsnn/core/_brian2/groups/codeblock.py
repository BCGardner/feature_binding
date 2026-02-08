from __future__ import annotations

from typing import Optional


class CodeBlock:
    def __init__(self, value: Optional[str | CodeBlock] = None) -> None:
        self.value = self._as_str(value)

    def __add__(self, other: Optional[str | CodeBlock]):
        return CodeBlock(self._as_str(self.value) + '\n' + self._as_str(other))

    def __repr__(self) -> str:
        return str(self.value)

    def __str__(self) -> str:
        return self._as_str(self)

    @property
    def value(self) -> Optional[str]:
        return self._value

    @value.setter
    def value(self, value: Optional[str]):
        self._value = self._parse(value)

    def _as_str(self, arg: Optional[str | CodeBlock]) -> str:
        if isinstance(arg, CodeBlock):
            arg = arg.value
        if isinstance(arg, str):
            return arg
        elif arg is None:
            return ""
        else:
            raise TypeError(f"invalid argument: '{arg}'")

    def _parse(self, value: Optional[str]) -> Optional[str]:
        value = self._as_str(value)
        result = "\n".join([line.strip() for line in value.split('\n') if line.strip()])
        if len(result):
            return result
        return None
