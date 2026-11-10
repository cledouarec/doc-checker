# Third-party notices

The code of doc-checker is licensed under the Apache License 2.0 (see
`LICENSE`). The wheel and the source distribution also carry data under the
Mozilla Public License 2.0, listed under "Bundled data". The libraries doc-checker
depends on are installed from PyPI next to it and are not redistributed in the
package; they are listed here with the licence each one declares in the
metadata of the version locked in `uv.lock`, and each licence text ships in
that library's own `dist-info` folder.

## Bundled data

| Data | Licence (SPDX) | Source |
|---|---|---|
| Grammalecte 7.7, French Hunspell dictionaries, "toutes variantes" (`fr-toutesvariantes.aff`, `fr-toutesvariantes.dic`, `README_dict_fr.txt`) | MPL-2.0 | <https://grammalecte.net/dic/hunspell-french-dictionaries-v7.7.zip> |

The files are in `src/doc_checker/spell/data/fr/` exactly as published
(unchanged, their SHA-256 sums are in
`src/doc_checker/spell/data/fr/SOURCE.txt`). The licence text is
`src/doc_checker/spell/data/fr/LICENSE`. The Source Code Form of the
dictionary is available from the archive above.

## Runtime dependencies

Installed with `pip install doc-checker`.

| Name | Licence (SPDX) | Project |
|---|---|---|
| click | BSD-3-Clause | <https://github.com/pallets/click/> |
| cssselect | BSD-3-Clause | <https://github.com/scrapy/cssselect> |
| Jinja2 | BSD-3-Clause | <https://github.com/pallets/jinja/> |
| MarkupSafe (via Jinja2) | BSD-3-Clause | <https://github.com/pallets/markupsafe/> |
| lxml | BSD-3-Clause | <https://github.com/lxml/lxml> |
| numpy | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | <https://github.com/numpy/numpy> |
| Pillow | MIT-CMU | <https://github.com/python-pillow/Pillow> |
| pypdf | BSD-3-Clause | <https://github.com/py-pdf/pypdf> |
| pypdfium2 | BSD-3-Clause, Apache-2.0 and "dependency licenses", as declared (the PDFium build licences are in its `dist-info`) | <https://github.com/pypdfium2-team/pypdfium2> |
| rapidfuzz | MIT | <https://github.com/rapidfuzz/RapidFuzz> |
| spylls | MPL-2.0 | <https://github.com/zverok/spylls> |

## pdf-tables extra

Installed with `pip install "doc-checker[pdf-tables]"`, on top of the runtime
dependencies.

| Name | Licence (SPDX) | Project |
|---|---|---|
| pdfplumber | MIT | <https://github.com/jsvine/pdfplumber> |
| pdfminer.six (via pdfplumber) | MIT | <https://github.com/pdfminer/pdfminer.six> |
| charset-normalizer (via pdfminer.six) | MIT | <https://github.com/jawah/charset_normalizer> |
| cryptography (via pdfminer.six) | Apache-2.0 OR BSD-3-Clause | <https://github.com/pyca/cryptography/> |
| cffi (via cryptography) | MIT-0 | <https://github.com/python-cffi/cffi> |
| pycparser (via cffi) | BSD-3-Clause | <https://github.com/eliben/pycparser> |

## Notes

- spylls is under MPL-2.0: its `LICENSE` file and its README say so, although
  its PyPI classifier says MIT. The licence file is what binds, so it is the
  one recorded here.
- The English dictionary that spylls bundles (`en_US`, from SCOWL/Hunspell)
  comes without a licence file. Its `.aff` file keeps the GNU LGPL-2.1
  notice of its phonetic table, which doc-checker does not use.
- The wheels of lxml, numpy, Pillow and pypdfium2 embed native libraries under
  their own licences. They are listed in the `dist-info` folder of each of
  these packages.
- colorama (BSD-3-Clause) is not a dependency of doc-checker: the locked
  click 8.5.0 does not require it. It enters the lock file only through
  pytest, on Windows, in the development group.
- pdfplumber's PyPI classifier and its `LICENSE.txt` agree on MIT. The
  licences above come from the metadata of the locked versions and from the
  licence files in their `dist-info` folders.
