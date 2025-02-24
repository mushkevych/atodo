import inspect
from os import path


def get_module_location() -> str:
    """
    Returns the absolute directory path of the module that calls this function.

    This function determines the location of the script/module that invokes it,
    rather than the location of the utility module where it is defined. It works
    by inspecting the caller's frame in the call stack.

    :returns: str: The absolute directory path of the caller module.
    """
    frame = inspect.currentframe()
    caller_frame = frame.f_back  # Get the caller's frame
    caller_file = inspect.getfile(caller_frame)  # Get caller's file path
    return path.dirname(path.abspath(caller_file))


def load_api_key(file_name: str) -> str:
    fqfp_token = path.abspath(path.join(get_module_location(), '..', file_name))
    with open(fqfp_token, 'r') as f:
        token = f.read().strip()
        # print(token)
        return token
