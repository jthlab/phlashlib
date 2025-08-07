from jaxtyping import install_import_hook

install_import_hook("phlashlib", "beartype.beartype")

from .loglik import loglik

___all__ = ["loglik"]
