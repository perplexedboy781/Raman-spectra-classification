import sys
from contextlib import contextmanager


@contextmanager
def redirect_stdout_to_file(file_path):
    with open(file_path, "w", encoding="utf-8") as file:
        original_stdout = sys.stdout
        sys.stdout = file
        try:
            yield
        finally:
            sys.stdout = original_stdout

# 使用上下文管理器
'''with redirect_stdout_to_file("output.txt"):
    print(model)
    print(opt)'''


