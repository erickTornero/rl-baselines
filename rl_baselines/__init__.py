__modules__ = {}
__version__ = "0.0.1"


def register(name: str):
    def decorator(cls):
        if name in __modules__:
            raise ValueError(
                f"Module {name} already exists! Names of extensions conflict!"
            )
        else:
            __modules__[name] = cls
        return cls

    return decorator

def find(name: str):
    if ":" in name:
        main_name, sub_name = name.split(":")
        if "," in sub_name:
            name_list = sub_name.split(",")
        else:
            name_list = [sub_name]
        name_list.append(main_name)
        NewClass = type(
            f"{main_name}.{sub_name}",
            tuple([__modules__[name] for name in name_list]),
            {},
        )
        return NewClass
    return __modules__[name]

from . import data, systems