import re
from typing import Iterable


def _get_uid(names: Iterable[str], startswith: str) -> str:
    def _extract_number(s):
        match = re.search(r'\d+', s)
        if match:
            return int(match.group())
        else:
            return 0
    pattern = re.compile(rf"^{startswith}(_\d+)?$")
    matches = [item for item in names if pattern.match(item)]
    if len(matches) > 0:
        max_num = max([_extract_number(s) for s in matches])
        return f'{startswith}_{max_num + 1}'
    return startswith
