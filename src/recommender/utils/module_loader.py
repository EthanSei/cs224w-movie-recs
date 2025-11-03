from importlib import import_module

def load_module(module_path):
    try:
        mod, obj = module_path.split(":")
        return getattr(import_module(mod), obj)
    except Exception as e:
        raise ImportError(f"Failed to load module {module_path}: {e}")