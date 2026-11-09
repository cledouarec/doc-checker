# Copyright 2026 Christophe Le Douarec
"""The bundled Hunspell dictionaries, read from bytes.

The French dictionary is Grammalecte 7.7 and the English one is the en_US
dictionary that spylls ships. Both are read with
``importlib.resources`` and decoded in memory, so no file stays open.
"""

import io
from functools import cache
from importlib.resources import files
from typing import Final

from spylls.hunspell import Dictionary, readers
from spylls.hunspell.readers.file_reader import BaseReader

DICTIONARY_LANGUAGES: Final = ("fr", "en")
FRENCH_DICTIONARY: Final = "fr-toutesvariantes"
_ENGLISH_DICTIONARY: Final = "en_US"
_DEFAULT_ENCODING: Final = "Windows-1252"


class _BytesReader(BaseReader):
    """A spylls reader over bytes already read, with no file to close."""

    def __init__(self, data: bytes, encoding: str = _DEFAULT_ENCODING) -> None:
        """Remember the bytes and start reading them.

        Args:
            data: The whole content of the dictionary file.
            encoding: The encoding the first lines are decoded with.
        """
        self._data = data
        super().__init__(self._decode(encoding))

    def reset_encoding(self, encoding: str) -> None:
        """Read again from the start, decoded with another encoding.

        Args:
            encoding: The encoding the affix file declares.
        """
        self.reset_io(self._decode(encoding))

    def _decode(self, encoding: str) -> io.StringIO:
        """Decode the bytes, keeping invalid bytes as surrogates.

        Args:
            encoding: The encoding to decode with.

        Returns:
            The text, ready to be read line by line.
        """
        return io.StringIO(
            self._data.decode(encoding, errors="surrogateescape"),
        )


@cache
def load_dictionary(lang: str) -> Dictionary:
    """Load the bundled dictionary of a language, once per process.

    Args:
        lang: ``"fr"`` or ``"en"``.

    Returns:
        The Hunspell dictionary.

    Raises:
        ValueError: If the language has no dictionary.
    """
    if lang == "fr":
        folder = files("doc_checker.spell") / "data" / "fr"
        name = FRENCH_DICTIONARY
    elif lang == "en":
        folder = files("spylls.hunspell") / "data" / "en"
        name = _ENGLISH_DICTIONARY
    else:
        message = f"no dictionary for {lang!r}"
        raise ValueError(message)
    aff_bytes = (folder / f"{name}.aff").read_bytes()
    dic_bytes = (folder / f"{name}.dic").read_bytes()
    aff, context = readers.read_aff(_BytesReader(aff_bytes))
    dic = readers.read_dic(
        _BytesReader(dic_bytes, context.encoding),
        aff=aff,
        context=context,
    )
    return Dictionary(aff, dic)
