# Doc Checker

## Overview

Compare and audit PDF and HTML documentation.

When a documentation moves to another generator (Sphinx to MkDocs, LaTeX to
another toolchain...), `doc-checker compare` checks that nothing was lost or
altered: missing sections and paragraphs, changed numbers or negations,
typos, which the dictionaries tell from corrections, lost accents or
ligatures, broken encodings. It also reports headings, tables, code and
lists that lose their kind, headings whose level changed, and links that
are lost, broken or pointed elsewhere. It compares the images too:
missing, added and replaced images, degraded images (resolution,
sharpness, SSIM, JPEG quality, colors, transparency), re-encoded images,
rasterized figures and lost alternative texts. When five findings or more of
one rule make the same change, such as the same lost accent or the same
missing block, they are reported as a single grouped finding; moved, split
or merged blocks and missing sections are never grouped. Layout differences
such as pagination, fonts, page chrome or typographic quotes are ignored.

`doc-checker audit` checks a single build, without a reference: broken
internal links, skipped heading levels, empty sections, unresolved
references, accessibility problems such as missing alternative texts,
images of a poor quality, and the text: spelling, typography, repeated
words and long sentences, in French and in English.

The current version compares two builds or audits one. `compare` takes
either two HTML builds, each a single page or a generated site directory,
or two PDF documents. Both inputs must have the same format: an HTML build
cannot be compared with a PDF. `audit` takes one of them.

## Installation

doc-checker needs Python 3.14 or later. Install the `doc-checker` command
from PyPI with [uv](https://docs.astral.sh/uv/), pipx or pip:

```bash
uv tool install doc-checker
pipx install doc-checker
pip install doc-checker
```

The optional `pdf-tables` extra reads the ruled tables of PDF documents, as
described in [PDF tables](#pdf-tables):

```bash
uv tool install "doc-checker[pdf-tables]"
```

The development version installs from the repository:

```bash
uv tool install git+https://github.com/cledouarec/doc-checker
```

## Usage

Compare a reference build with its new version:

```bash
doc-checker compare docs/_build/html site/ -o doc-checker-report
doc-checker compare old.pdf new.pdf
```

The command prints a summary and writes two reports in the output directory:

- `report.html` is a single file to open in a browser. It needs no network
  access. It has filters by severity, category and rule, a search box and a
  side-by-side view of the reference and new texts. The findings on images
  show thumbnails of the images, except SVGs, which are never rasterized.
- `report.md` lists the findings by severity, then by category, with the
  reference and new texts side by side. Changed code is shown as a diff. The
  thumbnails of the images are written in `report_assets/`, next to it.

`--format html,md` is the default. Use `--format md` to write only the
Markdown report.

Print the normalized blocks of a document, to tune the content selector of an
HTML build or to understand a finding:

```bash
doc-checker extract site/
doc-checker extract manual.pdf
```

The options of `compare`:

- `-o DIR`, `--output DIR`: directory of the reports, `doc-checker-report`
  by default.
- `-c FILE`, `--config FILE`: configuration file, as described in
  [Configuration](#configuration).
- `--lang auto|fr|en`: language of the documentation, detected by default.
- `--min-severity info|minor|major|critical`: lowest severity reported.
- `--content-selector CSS`: where the content is in each HTML page, as
  described in [HTML](#html).
- `--exclude GLOB`: skip site pages.
- `--disable RULE`: disable a rule, such as `text.case_changed` or `text.*`.
- `--ignore-moves`: do not report blocks that only changed place.
- `--no-images`: leave the images out of the extraction and the comparison.
- `--pdf-tables`: read the ruled tables of PDF documents as tables, as
  described in [PDF tables](#pdf-tables).
- `--format html,md`: report files to write, `html`, `md` or both.
- `--words FILE`: personal words that the dictionaries should know, one per
  line, as described in [Text checks](#text-checks). It can be repeated.

`extract` takes those that tell how to read a document: `--config` (`-c`),
`--lang`, `--content-selector`, `--exclude`, `--no-images` and
`--pdf-tables`.

### Typos

A word replaced by a close one, one letter apart, or two for a word of more
than four letters, is a typo, a word replaced or a typo fixed. When both
words are words of prose and both documents are in French or in English,
the dictionaries tell them apart. A new word that they do not know is a
typo, `text.typo` (major). A new word that they know, that the reference
uses or that is a personal word replaces the old one, `text.word_replaced`
(major), unless the old word is in lower case and unknown: the new word
then corrects it, `text.typo_fixed` (info), as `receive` corrects
`recieve`. The words of prose are the words that the audit spells: those
of headings, paragraphs, list items, tables, captions, quotes, admonitions
and footnotes, outside their inline code, URLs, e-mail addresses, paths
and options, that are neither acronyms nor words with a digit or a capital
inside. In code, formulas and the other blocks, the dictionaries are not
asked: a new word that the reference also uses in other places is a word
replaced, and any other is a typo, both major. A PDF does not mark its
inline code: an identifier set in a monospaced font inside a paragraph is
a word of prose there, so that `pcell` renamed `cell` is reported as a
typo fixed, an info.

The dictionaries are loaded at the first close word of prose: the English
one costs about 0.15 s and 40 MB. An English word that the English
dictionary does not know is looked up in the French one too, which is
then loaded: about 0.3 s and 85 MB more, once per run (measured on the
computer of [Performance](#performance)). The lookups stop after a work
budget, with a warning; the words left are judged without the
dictionaries.

### Structure and links

Heading numbers and chapter labels such as "Chapter 3" come from the
generator: they are set apart from the heading text and never compared, so
renumbered headings give no finding. The reports count them under
"Normalized differences". Heading levels are compared with each other, not
as absolute values: when every heading moves down by one level, nothing is
reported. A default admonition title such as "Note", or a chapter label,
found in one document only gives no finding.

A document numbers its headings when three of them at least, and half of
them, start with a number such as `2.1`. When only one document of a pair
does, the numbers of the other are set apart too, as long as three of its
headings at least start with one, so that the numbering of a build does
not depend on the share of its headings that have a number. Headings that
lose their numbers in a new build give no finding either. A numbered
heading with a very short title, such as "1.2.3 API", that becomes a
paragraph is reported as a missing block and an added one, not as a lost
heading: the paragraph keeps the number that the heading sets apart, and
the two texts are then too different to be paired.

A link is compared by the section it points to, not by its URL. Only the
links of matched blocks are compared, and only a link that worked in the
reference is reported: when it is lost, when it no longer works, or when it
points to another section or file. A page without content, such as a
download or an excluded page, is known by its name, so `api/foo.html` and
`api/foo/` are one page. Absolute links under the base URL of a site are
recognized as internal links. The links of a whole site are not checked by
`compare`; `audit` checks them.

The rules are `structure.heading_lost`, `structure.table_lost` and
`structure.code_lost` (major), `structure.list_lost` (minor),
`structure.kind_changed` (info), `structure.heading_level` (minor),
`link.broken` (critical), and `link.lost` and `link.target_changed` (major).
A group of findings has its own identifier, which can be ignored like any
other finding. An ignored group stays ignored when fewer than five of its
findings are left, after a partial fix for instance.

### Images and figures

In HTML, the images are the `img` elements, read from the largest candidate
of their `srcset`, else from their `src`, and the inline `svg` elements.
`data:` URLs are decoded. Icons, which are SVGs whose sides are 48 pixels or
less, emoji images and the images of formulas are ignored. A file is only
read under the root of the site. External images, with an absolute URL, are
never fetched, and never judged by their content.

In PDF, the images are the raster images of the pages and their vector
figures, which PDFium renders: the forms that pdflatex writes for an
included PDF or SVG, and the groups of shapes drawn on the page itself, as
LibreOffice draws them. A form is a figure when it holds a path. A group of
shapes at most 6 points apart is a figure when it shows a drawing: a curve,
or filled shapes of two colors or more. In both cases, the text inside must
read like labels: short lines, none wider than half the figure, 200
characters at most, and, in a group, no line inside a shape that covers
half the group. Boxes of text, such as notes, framed boxes, tables and
listings, keep their text. A figure has at least 36 points on each side and
covers less than half the page. The labels of a figure leave the compared
text.

The images are matched by their content first, then by their caption,
alternative text, file name or place. A pair of raster images is judged on
its pixels, and the resolution is compared in pixels, never in DPI. The new
image is degraded when its resolution falls below 0.7 of the reference, its
SSIM below 0.90 or its sharpness below 0.6, when its JPEG quality drops by
20 or more, or when it loses its colors or its transparency. A new image
that is larger, sharper or of a higher JPEG quality is never reported as
degraded for its SSIM or its sharpness. Between two PDFs, a lower
resolution is not reported while the new image keeps 300 DPI or more. A
change without such a loss is a re-encoding, an info. An SVG is never
rasterized: it is compared by its source, then by its texts, and the same
texts with another source are a re-encoding. Two vector figures of PDFs
with the same content give no finding. A vector figure or an SVG that
becomes a raster image is rasterized, and the reverse is vectorized. Two
images of different contents that share their context are replaced.

Caption labels and numbers, such as "Figure 3:", come from the generator
and are never compared, as heading numbers are. An image of fewer than 4
pixels on a side, such as a spacer or a rule drawn as an image, is ignored,
in HTML and in PDF. An image found on half the pages of a site, or on the
`repeat_share` of the pages of a PDF, and on three pages at least, is page
chrome, such as a logo. It is left out of both documents, whichever one
repeats it. The alternative text is only compared in HTML: its loss is
reported, not its change. The reports count the caption numbers, icons,
tiny images, image chrome and figure labels set apart under "Normalized
differences".

The rules are `image.missing` (critical), `image.replaced` and
`image.degraded` (major, critical when the resolution falls below 0.4),
`image.added`, `image.rasterized` and `image.alt_lost` (minor), and
`image.reencoded` and `image.vectorized` (info). `--no-images` leaves the
images out, but the labels of the vector figures still leave the text, so
that the compared text does not depend on the option.

#### Known limits

- A JPEG of more than 64 scans, or of more than 4,096 markers, is not
  read: it is reported as too large. Real JPEG files have far fewer: 14
  scans at most among 667 measured.
- The copies of an image used several times are paired in their order:
  when one copy is removed or changed, the finding may name another copy,
  at another place.
- A group of shapes is not found as a figure when it is a diagram drawn on
  a background panel or inside one big frame, or when it has only straight
  lines and shapes of one fill color. Its labels then stay in the text. The
  shapes of a drawing more than 6 points away from the others, and the
  labels outside its shapes, such as a title above it, are left out of its
  figure too.
- A vector figure drawn as one form in one PDF and as loose shapes in the
  other may not be found as the same figure. A post-processor that
  flattens forms, such as Ghostscript, does this, and so does a figure
  whose parts do not touch, which pdflatex includes as one form and
  LibreOffice draws as separate shapes. The figure is then reported
  missing, or replaced when its place pairs it with what the other PDF
  found, and its labels show as added or missing text.
- At most 64 vector figures are rendered on a page. The others are not
  compared by their content, and a warning says so, but their labels still
  leave the text.
- A sparse page included whole, as pdfpages includes a table of contents, a
  chapter opener or a slide, may be taken for a figure and lose its text.
  So may a callout drawn as a bar with an icon but no background.
- Only the filter chains that generators write are read in PDF image
  streams. Other chains, LZW, an image wrapped in ASCII85 or ASCIIHex of
  more than 16 MB, and an image of more than four color components are
  reported as unreadable.
- An image that cannot be read, because its file is missing, outside the
  site or too large, or because it cannot be decoded, is only reported in a
  warning. A new image that cannot be read where the reference one could is
  reported missing.
- An image replaced by an external one, or an external image whose URL
  changes, gives no finding when the two share their context.
- The alternative text of PDF images is not compared; `audit` checks it in
  the tags of a tagged PDF.

### Audit

Audit a single build, an HTML build or a PDF document:

```bash
doc-checker audit site/ -o audit-report
doc-checker audit manual.pdf --profile print
doc-checker audit site/ --checks structure,a11y --min-severity minor
```

The command prints a summary and writes the same reports as `compare`,
`report.html` and `report.md`, with one column, "Document", instead of the
reference and the new texts. The part of an excerpt that a finding is
about, such as the text of a vague link or a `??`, is highlighted. The
reports count the normalizations of the extraction under "Normalizations".
The findings are grouped, identified and ignored as those of `compare`,
but `[ignore] patterns` apply to `compare` only: the audit checks the texts
as they are. No audit rule is critical by default.

The audit takes the options of `compare` except `--ignore-moves`:
`--output` (`-o`), `--format`, `--min-severity`, `--disable`, `--words`,
`--config` (`-c`), `--lang`, `--content-selector`, `--exclude`,
`--no-images` and `--pdf-tables`. It has three of its own:

- `--checks LIST`: comma separated families of checks to run, all of them
  by default: `structure` (headings, references, captions, pages and
  links), `a11y` (accessibility), `images`, and `text` (spelling,
  typography and style, see [Text checks](#text-checks)).
- `--profile screen|print`: resolution expected of the images of a PDF,
  `screen` by default.
- `--emit-wordlist FILE`: write the words that the dictionaries do not
  know to `FILE`, one per line.

`--no-images` leaves out the `images` family, and the alternative texts of
the HTML images with it. The alternative texts of a tagged PDF, read from
its tags, are still checked.

| Rule | Family | Severity | Reports |
|---|---|---|---|
| `structure.heading_skipped` | structure | minor | A heading more than one level deeper than the heading before it, in an HTML page or in the tags of a PDF |
| `structure.empty_section` | structure | minor | A heading followed by a heading of its level or above, or by the end of its page, with no content between them |
| `structure.duplicate_heading` | structure | minor | Two sibling headings of a page with the same title, case ignored |
| `structure.unresolved_reference` | structure | major | A lone `??` or `[?]`; in HTML, also a literal `[text][label]` or a Sphinx or docutils reference without a target |
| `structure.figure_caption` | structure | info | A figure without a caption, in a document that captions at least one figure |
| `structure.table_caption` | structure | info | A table without a caption, in a document that captions at least one table |
| `structure.orphan_page` | structure | minor | A page of a site, other than the first one, that no other page links to |
| `link.page_missing` | structure | major | An internal link to a page that does not exist |
| `link.anchor_missing` | structure | major | An internal link to an anchor that its page does not have |
| `link.file_missing` | structure | major | A link to a file of the site that does not exist, or a link of a PDF that opens a missing local file |
| `link.destination_invalid` | structure | major | A link of a PDF that leads to no page of the document, or that has neither a destination nor an action |
| `a11y.alt_missing` | a11y | major | An HTML image without `alt` nor `aria-label`, or a figure of the tags of a PDF without `/Alt` nor `/ActualText` |
| `a11y.alt_filename` | a11y | minor | An alternative text that is a file name, such as `_images/setup.png` |
| `a11y.svg_title` | a11y | minor | An inline SVG without a `title` nor an `aria-label` |
| `a11y.lang_missing` | a11y | major | An HTML page or a PDF that does not declare its language as a language tag, such as `en-US` |
| `a11y.lang_mismatch` | a11y | minor | A document that declares French or English but whose text plainly reads as the other language |
| `a11y.table_headers` | a11y | minor | A table without a header cell, in the content of an HTML page or in the tags of a PDF; layout tables are left out |
| `a11y.pdf_untagged` | a11y | major | A PDF without tags (structure tree) |
| `a11y.title_missing` | a11y | minor | An HTML page without a title, or a PDF without a title in its metadata |
| `a11y.unmapped_chars` | a11y | minor, major from `unmapped_major` of the characters | Characters of a PDF without a Unicode mapping, in one finding for the document |
| `a11y.scanned_page` | a11y | major | A page of a PDF that is an image without text |
| `a11y.link_empty` | a11y | major | A link of an HTML page without text, label or named image |
| `a11y.link_text` | a11y | minor | A link whose text does not tell where it leads, such as "click here", "read more" or "lire la suite" |
| `image.broken` | images | major | An image whose file is missing or outside the site, or that cannot be decoded |
| `image.heavy` | images | minor, major beyond `heavy_major` | An image file of more than `heavy_minor` bytes |
| `image.low_dpi` | images | minor, major below `dpi_major` (`print_dpi_major`) | A raster image of a PDF whose resolution is below `dpi_minor` (`print_dpi_minor` with the `print` profile) |
| `image.upscaled` | images | minor | An HTML image that its `width` and `height` show more than `upscale` times its size |
| `image.jpeg_quality` | images | minor, major below `jpeg_quality_major` | A JPEG whose estimated quality is below `jpeg_quality_minor` |
| `image.blurry` | images | minor | A raster image whose sharpness is below `blur` |
| `image.blocky` | images | minor | A raster image that shows the 8x8 blocks of a strong JPEG compression, such as a JPEG saved again as PNG |
| `image.pixelated` | images | minor | A raster image whose rows and columns repeat, as when it is enlarged without smoothing |
| `image.photo_png` | images | info | A photograph stored as a PNG of 100 KiB or more, which a JPEG would hold in fewer bytes |

The rules of the `text` family are in [Text checks](#text-checks).

In HTML, the checks read the content of each page, as `compare` does; the
links of the navigation only tell which pages are linked to. Each internal
link of the content is checked against the files of the site and the
identifiers of its target page. A link to a file without suffix, such as
`LICENSE`, leads to that file. The `top` fragment and text fragments
(`#:~:text=`) are never checked as anchors. An empty alternative text marks
a decorative image, as does the `presentation` or `none` role. An embedded
medium, such as a video or a frame, a Sphinx toctree and, under
`--no-images`, an image are content of their section.

In a PDF, the sections are those of the outline (bookmarks): a heading
found by its style is content, and two headings are siblings when their
bookmarks are. Skipped levels, alternative texts and table headers are read
from the tags; an untagged PDF gives `a11y.pdf_untagged` instead. Each link
annotation is checked: its destination must lead to a page of the
document, and a link that opens a local file must find it next to the PDF.
A URI is never checked.

The weight, the JPEG quality and the pixels of an image file are reported
once, at its first use; a broken image, a low resolution and an upscaled
image, at each use. A raster image of a PDF is only judged by its
resolution when it is shown on 36 points or more. An image that is too
large to be read, an external image and a vector figure of a PDF are never
reported broken: a browser or a viewer may still show them. Neither is an
AVIF or JPEG XL image, which browsers show though doc-checker cannot decode
it. `image.blocky` is not reported for an image that already has an
`image.jpeg_quality` finding, even when that finding is disabled or
ignored.

The families and the profile can also be set in the `[general]` section of
the configuration, and the thresholds in `[thresholds.audit]`. These are
the defaults:

```toml
[general]
checks = ["images", "structure", "a11y", "text"]
profile = "screen"          # or "print"

[thresholds.audit]
dpi_major = 96              # screen profile: major below, in dots per inch
dpi_minor = 150             # screen profile: minor below
print_dpi_major = 150       # print profile: major below
print_dpi_minor = 300       # print profile: minor below
heavy_minor = 1_000_000     # image file, in bytes: minor beyond
heavy_major = 3_000_000     # image file, in bytes: major beyond
jpeg_quality_minor = 50     # estimated JPEG quality: minor below
jpeg_quality_major = 25     # estimated JPEG quality: major below
upscale = 1.1               # shown size over image size: upscaled beyond
blur = 0.10                 # sharpness: blurry below
blockiness = 2.0            # JPEG blocks: blocky from
unmapped_major = 0.01       # share of unmapped PDF characters: major from
long_sentence_minor = 40    # words in a sentence: minor beyond
long_sentence_major = 60    # words in a sentence: major beyond
```

A major limit must be stricter than its minor one, `upscale` and
`blockiness` above 1, `blur` and `unmapped_major` at most 1, and the JPEG
qualities between 1 and 100. An unknown family or an invalid threshold
stops the command with exit code 2. `[severity]` changes the severity of an
audit rule as of any other.

#### Text checks

The family `text` checks the spelling, the typography and the style of the
prose, in French and in English. The prose is the text of the headings,
paragraphs, list items, tables, captions, quotes, admonitions and
footnotes. Code blocks, formulas, tables of contents, indexes and the
alternative texts of images are never checked, nor is a block that looks
like code, where a quarter of the chunks or more hold a mark of code such
as `\`, `{`, `=`, `$`, `::` or `->`. Inside the prose, the checks leave out
the inline code, the URLs, the e-mail addresses and the chunks that look
like code: a chunk with a mark of code such as `/`, `\`, `_`, `=`, `<`,
`[`, `{`, `$` or `::`, a dot inside a word (`conf.py`), a leading dot
(`.gitignore`) or a leading hyphen (`--verbose`). The inline code of an
HTML page is the text of its `code`, `kbd`, `samp`, `tt` and `var`
elements, and in a tex4ht page, of the text that it sets in a typewriter
font, such as `span.cmtt-10`. A PDF does not mark its inline code, but it
tells the words that it sets in a monospaced font: the spelling never
reports them, wherever they are, and the typography does not judge a number
glued to its unit, nor a blank or a missing space before a mark after a
word, when that number and unit or that word is set so somewhere in the
document, as `1cm` in `xshift=1cm` or `listings` before a comma.

A word is spelled when it has two letters or more, 40 characters and 5
parts between hyphens at most, and only Latin letters, apostrophes and
hyphens. A word with a digit, such as `x86` or `2e`, an acronym, such as
`API` or `API's`, and a word with a capital inside, such as `GitHub` or
`camelCase`, are never spelled.

The French dictionary is Grammalecte 7.7 "toutes variantes", which knows
both spellings of the 1990 reform, such as `événement` and `évènement`. It
is published under the MPL-2.0 and shipped unchanged with doc-checker;
`src/doc_checker/spell/data/fr/SOURCE.txt` gives its source and its
checksums. The English dictionary is the en_US dictionary that comes with
spylls, the port of Hunspell that doc-checker uses, published under the
MPL-2.0; the dictionary has no licence file of its own, and its affix file
keeps the LGPL-2.1 notice of its phonetic rules. Both dictionaries come
with the packages: nothing is downloaded, and no dictionary of the system
is read. A word that the dictionary of its language does not know
is known when the dictionary of the other language knows it: technical
French quotes English words, such as `authentication`, and English quotes
French ones, such as `café`. In French, a word after an elided word, such
as `l'upload` or `d'authentication`, is known when the word without its
elided start is a personal word or an English word. In English, a
possessive (`user's`), a British spelling (`colour`, `organise`, `centre`,
`catalogue`, `defence`, `travelled`, `acknowledgement`, `recognisable`)
and a known word after a common prefix, such as `subcommand` or
`unbuffered`, are known. In both languages, a word cut by a hyphen at
the end of a line, such as `connec-tion`, is known when it is known
without its hyphens, and a hyphenated word when each of its parts is known
or code, such as `config-based`.

Each block is checked in the language of the text of the document: the
language of `--lang` or `[general] lang`, else the one that its prose
plainly reads as, else the one that it declares. A paragraph, a list item,
a caption, a quote, an admonition or a footnote that plainly reads as the
other language is checked in that language, and reported as
`text.block_language`. In a document of another language, or whose
language cannot be told, the checks that depend on the language read only
the blocks that plainly read as French or as English, and a warning says
so; the units, the doubled punctuation and the long sentences are judged
in every block. The language guard catches a document checked in the
wrong language: once 200 words are spelled, if more than 25 % of them are
unknown, the findings that depend on the language, all but those of
`text.unit_space`, `text.double_punctuation` and `text.long_sentence`, are
left out, the word list stays empty, and a warning suggests `--lang`, or a
language without dictionary. A block of 20 spelled words or more, more
than half of them unknown, such as placeholder text (lorem ipsum) or a list
of package names, gives no spelling finding and none of its words go to
the word list, though they count for the jargon of the document; a warning
counts such blocks.

| Rule | Severity | Reports |
|---|---|---|
| `text.unknown_word` | minor | A word that the dictionaries do not know, that does not start with a capital, seen once or twice in the document |
| `text.unknown_name` | info | An unknown word that starts with a capital, seen once or twice: it may be a name |
| `text.jargon` | info | An unknown word seen 3 times or more in the document, whatever its case: the vocabulary of the document |
| `text.space_missing` | minor | In French, no space before `;`, `:`, `!`, `?` or `»`, or after `«` |
| `text.space_breakable` | info | In French, in HTML, a space that can break the line at those places, where a non-breaking space is due |
| `text.space_before_punctuation` | minor | In English, a blank before `,`, `;`, `:`, `!` or `?` |
| `text.quotes` | info | French quoted with `"…"` or `“…”`, or English quoted with `«…»` |
| `text.ordinal` | minor | A French ordinal written with too many letters, such as `2ème` for `2e` or `1ère` for `1re`, or an English ordinal with a wrong ending, such as `21th` |
| `text.unit_space` | minor | A number glued to a unit, such as `4GB`, `100ms` or, in French, `4Go` |
| `text.double_punctuation` | minor | Two marks of punctuation after a word where one is meant: `,,`, `;;`, `!!`, `..`, `,.`, `;.` or `:.` |
| `text.repeated_word` | minor | A word of the dictionary written twice in a row, such as "the the" |
| `text.long_sentence` | minor, major beyond `long_sentence_major` | A sentence of more than `long_sentence_minor` words |
| `text.block_language` | info | A block that plainly reads as the other language: French in an English document, or English in a French one |

The unknown words are counted over the whole document, without case, with
their possessives and their elided forms: a word seen three times or more
is the vocabulary of the document, a word that starts with a capital may
be a name, and only the others are minor findings. The findings of these
three rules are grouped by word from two occurrences, and the other
findings from five, as usual. A finding of the text highlights the word,
the punctuation or the sentence that it is about; a block in another
language is shown whole. The reports count the words spelled under
"Words", and the summary of the terminal as `words`.

The units are those of length, mass, time, frequency, storage, power,
energy, charge and sound, such as `km`, `mg`, `ms`, `GHz`, `MB`, `GiB`,
`kW`, `kWh`, `mAh` and `dB`, and in French text only the French storage
units, such as `Go` and `Mio`: in English, `10To be` is a footnote number
before a word. A unit is never `px`, `pt` or `em`, which are values of CSS
or LaTeX, nor a unit of one letter, as in `4G` or `3D`.

A word is repeated when the second one has the case of the first or is in
lower case, unless the language repeats it on purpose, as in `vous vous`,
`nous nous`, `that that` and `had had`, and never across a line break. A
run of three or more, as in "really really really", is meant, and not
reported.

A sentence ends with `.`, `!`, `?` or `…` before a blank or the end of its
block, but not after an abbreviation such as `e.g.`, `etc.` or `cf.`, nor
after an initial, nor before a lower-case letter. Its words and numbers are
counted, outside the inline code, URLs and code: beyond
`long_sentence_minor` words, 40 by default, the sentence is a minor
finding, and beyond `long_sentence_major`, 60, a major one. The sentences
of headings and tables are not measured.

Personal words are words that the dictionaries should know, such as the
names of a product and of its commands. They come from the `[spelling]`
section of the configuration and from `--words FILE`, which can be
repeated, for `audit` as for `compare`:

```toml
[spelling]
words = ["kubectl", "Orbit"]
files = ["words.txt"]       # relative to the configuration file
```

A file of words is UTF-8 text with one word per line; the blanks around a
word, the blank lines and the lines that start with `#` are ignored. A file
that cannot be read, that is not UTF-8 or that has two words on a line
stops the command with exit code 2. A personal word is known whatever its
case, and so is its possessive.

`--emit-wordlist FILE` writes the words that the dictionaries do not know
to `FILE`, each once, as first written, sorted without case, one per line
in UTF-8. The file is written before the reports, even when it is empty;
it may be in the output folder, which is then created first, and a file
whose folder does not exist stops the command before the audit.
It holds every unknown word, whatever its rule, `--min-severity`,
`--disable` and `[ignore]`; the language guard leaves it empty. The option
needs the family `text`. The list has the format of a file of words, but
it holds the misspellings along with the vocabulary: review it before
passing it to `--words`. A first audit gives the list:

```bash
doc-checker audit site/ --emit-wordlist words.txt
# Fix the misspellings in the documentation, remove them from words.txt,
# keep the vocabulary, then:
doc-checker audit site/ --words words.txt
```

#### Known limits

- The tags of a PDF serve the audit only: the reading order and the
  headings of a tagged PDF still come from its layout, its outline and its
  styles, for `compare` as for `audit`.
- A skipped heading level is only found in the tags of a PDF, never in an
  untagged one. Empty sections and identical sibling headings of a PDF are
  only found among its bookmarks. A bookmark is matched with the first block
  of its page that holds its title: a list of the sections to come, or a
  table of contents whose page numbers are set apart, may take the place of
  a heading, whose section is then reported empty.
- The tags of an AES-encrypted PDF are not read unless the optional
  `cryptography` package, which the `pdf-tables` extra installs, or
  `pycryptodome`, is installed, nor those of a
  damaged PDF that a strict reading refuses; a warning says so. The tags of
  a very dense structure tree, beyond some 550 pages of tagged LaTeX, are
  read in part, with a warning.
- Under `--no-images`, a section of a PDF whose only content is an image
  is reported empty.
- The role maps of the PDF 2.0 namespaces are not followed: an element
  mapped to a heading, a figure or a table only through a namespace is not
  recognized.
- fpdf2 2.9 tags each section title as a figure without alternative text:
  each one is reported.
- A link of a PDF that opens a local file is only checked when its path is
  relative and stays inside the folder of the PDF; on macOS, a path that
  holds a `:` is never checked. On a page with thousands of links, the
  text of the last ones may be left unread, with a warning; at most 10,000
  links are read on a page.
- pdfTeX replaces the destination of a `\hyperlink` to a missing target
  with a fixed one: the link then leads to a page and is not reported.
- A lone `??` is taken for an unresolved reference, even a quoted one
  ("?? is used for untracked paths"); a `??` glued to a word, a `?` or an
  `=` is not. A `[text][label]` whose text or label holds `<`, `>` or `/`,
  or whose text is a character class such as `0-9a-f`, is no reference;
  so is a pair such as `[x-y][z]`, read as a class.
  In HTML, one inside inline code is not reported, but in a PDF a text that
  quotes the `??` operator is, even in a monospaced font, which is not told
  apart from the text around it. `[ignore] patterns` do not silence it.
- A link is highlighted in its block by its text: when the block repeats
  that text after the link, outside any link, the repetition may be
  highlighted instead.
- A page of headings and images only, such as the empty `tags/` and
  `categories/` pages of Hugo, is never an orphan.
- The anchor of a link to a page that exists but is not read, such as an
  excluded `genindex.html`, is not checked. External links are never
  checked: the audit needs no network access.
- The identifier of a finding in a PDF leaves out its page, so that it
  survives a new pagination. Findings of one rule that quote the same text
  on different pages, such as several tagged figures without alternative
  text, share it, and one entry of `[ignore] findings` ignores them all.
- An image or an inline SVG named only by `aria-labelledby`, and an image
  named only by its `title`, are reported without alternative text. An SVG
  given as a `data:` URL to an `img` without alternative text is reported
  as an SVG without title, a minor finding.
- An image enlarged by CSS, such as with `style="width: 800px"`, is not
  reported as upscaled: only its `width` and `height` attributes count.
- A PNG with a bad checksum in a metadata chunk is reported broken, though
  browsers show it. The pixels and the resolution of AVIF and JPEG XL
  images are not judged.
- The images of the page chrome, found on half the pages of a site or on
  the `repeat_share` of the pages of a PDF, are left out at extraction and
  never audited.
- The size of a `data:` URL too long to be decoded is estimated from its
  length.
- JPEG blocks saved as PNG in a screenshot whose edges fall on the 8-pixel
  grid are not reported as blocky.
- In a PDF, a missing space before `;`, `!` and `?` is not reported: the
  thin space that LaTeX sets there in French is lost when the text is
  extracted, so that a PDF does not tell it from no space. A breakable
  space where a non-breaking one is due is only reported in HTML.
- In a PDF, a word set in a monospaced font anywhere in the document is
  never reported as unknown, even where it is prose, and a number and unit
  or a word before a mark that is set so somewhere is not judged by the
  typography: a real `4GB` or `word ,` of the prose is missed when the
  same text is also code. The other checks read the inline code of a PDF
  as prose: a word written twice, a doubled mark or a long sentence inside
  it is reported.
- A misspelled word at the start of a sentence has a capital: it is
  reported as a possible name, `text.unknown_name`, an info.
- The English dictionary is small: it misses many technical words, such
  as `namespace`, `filesystem` or `async`, which are reported. Run a first
  audit with `--emit-wordlist`, then pass the reviewed list to `--words`.
- The other language and the English prefixes let some misspellings pass:
  `untill` and `postion` are known through `un` and `post`, and `exemple`
  in English through French. No correction is ever suggested.
- Only French and English have dictionaries. In a document of another
  language, the checks that depend on the language read only the blocks
  that plainly read as one of them, with a warning, and a document of
  which more than 25 % of the spelled words are unknown loses the findings
  that depend on the language, with a warning. A block of 20 spelled words
  or more, more than half of them unknown, loses its spelling findings,
  even when it is a dense table of real words that the dictionaries miss.
- A word written twice on purpose is reported, such as two labels of a
  figure that stayed in the text, or a repetition inside the inline code of
  a PDF, even of a word set in a monospaced font: common words such as
  `the` are set so somewhere in most manuals. A run of three or more is
  not reported.
- Only the sentences that end with `.`, `!`, `?` or `…` are measured: a
  long sentence without final punctuation is not reported.
- Documents are not trusted: the spelling looks each distinct word up once,
  and stops after a work budget, with a warning; the words left are not
  spelled, nor counted among the words spelled, and a word of them written
  twice is not reported. A hostile document of countless odd words spends
  the budget in about 27 s, while the 259 pages of the git manual use 37 %
  of it, and the 1,323 pages of the PGF manual 19 %. A word that cannot be
  looked up is taken as known, with a warning.
- Each rule of the text gives at most 10,000 findings, the first ones,
  with a warning.
- On a long HTML site, the checks of the text may take longer than the
  extraction: 5.2 s against 2.5 s for the 259 pages of the git manual, on
  the computer of [Performance](#performance). On a PDF, they take a
  fraction of it: 3.6 s against 37 s for the 1,323 pages of the PGF
  manual.

### HTML

An HTML build is a single page or the directory of a generated site. The
pages of a site are its `.html` and `.htm` files, except those that the
`exclude` patterns of the `[html]` section and `--exclude` leave out: by
default, generated pages and folders such as `genindex.html`,
`search.html`, `404.html`, `_static/` and `_sources/`. The pages are read
in the order of the navigation of the home page, `index.html`, then in the
order of their paths; `page_order = "path"` in the `[html]` section
follows the paths only.

Each page is read with the profile of its generator, which tells where its
content is, what to leave out of it and where the navigation is. The
profile is detected from the markup of each page, as the first of these
that matches:

| Profile | Generator | Left out of the content |
|---|---|---|
| `sphinx-rtd` | Sphinx with the Read the Docs theme, MkDocs with the `readthedocs` theme | previous and next buttons |
| `sphinx-pydata` | Sphinx with the PyData theme | previous and next links |
| `mkdocs-material` | Material for MkDocs | edit button, source file dates, feedback, tags, copy buttons |
| `sphinx` | Sphinx with another theme, such as Alabaster | edit link, related links, sidebar |
| `docusaurus` | Docusaurus | tables of contents, page footer, version badge |
| `vitepress` | VitePress | language labels of the code blocks |
| `antora` | Antora | toolbar, table of contents |
| `docsy` | Hugo with the Docsy theme | date of the last change, section index, feedback |
| `texinfo` | texinfo (`makeinfo --html`, `texi2any`) | navigation panels, menus, tables of contents, indexes, `¶` links |
| `tex4ht` | tex4ht (`make4ht`, `htlatex`) | navigation links, table of contents |
| `generic` | any other, such as Quarto, mdBook, Hugo, Jekyll, pandoc or MkDocs with its default theme | nothing more |

The Sphinx profiles also leave out the `contents` boxes. Every profile
leaves out the scripts, styles, buttons, `nav` elements, hidden elements,
toctrees, permalinks of the headings, line numbers and footnote back-links.
The `generic` profile reads the first `main`, `[role=main]`, `article`,
`#content` or `div.content` element. When a page has no content element,
as with the `texinfo` and `tex4ht` profiles, its content is its `body`,
without its `header`, `footer` and `aside` elements. The reports name the
profile of most pages under "Generator".

`profile` in the `[html]` section forces a profile on every page, `auto` by
default; an unknown name stops the command with exit code 2.
`--content-selector`, or `content_selector` in the `[html]` section, is a
CSS selector of the content tried first on each page, before the profile's.
`strip_selectors` lists the CSS selectors of more elements to leave out.
`extract` shows what is read.

The titles of admonitions, such as the `p.admonition-title` of Sphinx and
MkDocs, the titles of Docusaurus, VitePress, Docsy and texinfo, and those
of Asciidoctor, Quarto callouts and mdBook, are left out when they are
default titles, such as "Note", "Warning" or "Remarque", in English or in
French: one generator may write them where another does not. Any other
title is read as a paragraph, never as a heading. An Asciidoctor
admonition, which Antora and the git manual write as a table, is read as
its content, and `a11y.table_headers` takes it for a layout table.

#### Known limits

- A page of more than 100 MB is not read: reading a page takes about 14
  times its size in memory. The command stops with exit code 3; leave the
  page out with `--exclude`.
- The `docsy` profile follows the documented markup of the theme, without
  a real build to check it on, and the `antora` profile the markup of the
  default UI. The chrome that a profile misses, or that Quarto or mdBook
  set inside `main`, shows in `extract`; `--content-selector` and
  `strip_selectors` remove it.
- texinfo declares the language of its pages on `body` instead of `html`:
  `audit` reports each page with `a11y.lang_missing`, as accessibility
  checkers do, though the language of its text is still detected.

### PDF

PDF documents are compared on their text and their images.
Running headers, footers and page numbers are ignored: they are looked for
in the top and bottom bands of each page and on its first and last lines,
and a number is only taken for a page number when it follows the page
labels or the page order. Line-end hyphenation, columns and page breaks are
ignored too: a paragraph split across two columns or two pages is read as
one. Landscape pages are read as they are displayed; text set at another
angle, such as a diagonal watermark, is left out and reported in a warning.
Tables of contents and indexes are listed by `extract` but only compared when
`compare_toc` is set in the `[pdf]` section of the configuration. Scanned
pages are reported in a warning but not read, as there is no OCR. Tables are
compared as text, unless the `pdf-tables` extra reads them, as described in
[PDF tables](#pdf-tables). The report links to the pages, such as
`manual.pdf#page=12`. The `--content-selector` and `--exclude` options only
apply to HTML.

#### Known limits

PDF files do not say where paragraphs and tables begin, so the reading
guesses from the layout:

- Paragraphs are told apart by the space between them or by the indent of
  their first line. Without either, consecutive paragraphs are read as one,
  and a paragraph that ends without punctuation stays joined to the next.
- A quote or a verse indented as a block, without space around it, may be
  cut after a first line that ends a sentence, and so may the entries of a
  list with hanging indents, such as a bibliography, that mostly fit on two
  lines and never wrap in the middle of a sentence. The cuts depend on the
  width of the lines, so that two PDFs may differ there.
- A caption with no space between it and the paragraph after it, as
  LibreOffice sets it, is read with that paragraph, and two captions on
  either side of a page break may be read as one, while pdflatex, which
  floats its figures, sets its captions apart. Between such PDFs, these
  captions are reported missing.
- Tables are read as text, row by row or column by column depending on
  their layout. A header row repeated on each page may be taken for a
  running header in one document only. The `pdf-tables` extra lifts both
  limits for ruled tables.
- The kinds of blocks (headings, tables, code, lists) are guessed from the
  layout. A change of kind or level between two PDFs is therefore only an
  info, except for headings that come from the outline (bookmarks). A
  partial outline is completed from the style of the headings.
- The links of PDF files are not compared; `audit` checks them.
- Footnote markers are joined to the word before them, so two generators
  that number footnotes differently produce findings.
- Some generators leave spaces inside monospaced code. `extract` shows
  them, but they never change the comparison.
- The bands searched for running headers and footers cover 8 % of the page
  height (`header_band`, `footer_band`); the first and last lines of each
  page are searched up to 20 %. Raise the bands when a header or footer of
  several lines sits further from the edge. Lower them when body text runs
  close to the edge, as with narrow margins, and text repeated there
  disappears. With the `pdf-tables` extra, the rows of a ruled table are
  never taken for a header or a footer.
- Beyond the bands, the first or last line of a page is only taken for a
  header or footer when space sets it apart from the text, and only as a
  whole: if one of its parts is not recognized, the whole line stays in
  the text. So the running header of a LaTeX `book`, which sets the page
  number and the running title on one line, such as "10 CHAPTER 3. …",
  stays in the text when it sits beyond the bands, joined to the paragraph
  that the page continues: against a PDF of another generator, that
  paragraph is reported missing, a critical finding, on each such page.
- PDFium, which reads the PDF, decodes the content and the images of a
  page whole, among them the images that it only meets while it renders
  the page: soft masks, patterns, Type 3 glyphs, group masks and color
  profiles. A crafted PDF can make it spend gigabytes of memory, or
  seconds, on one page, whatever the options: 3.7 GB for a file of 5 MB,
  and 15 s on one page, were measured. Only a separate process with limits
  would bound it.

#### PDF tables

The optional `pdf-tables` extra reads the ruled tables of PDF documents as
tables. It installs pdfplumber, and it is used only when asked for, with
`--pdf-tables` on `extract`, `compare` and `audit`, or with `tables = true`
in the `[pdf]` section of the configuration. Asking for it without the extra
is a configuration error, whatever the documents. In a comparison, it
applies to both documents.

```bash
pip install 'doc-checker[pdf-tables]'
doc-checker compare old.pdf new.pdf --pdf-tables
```

A table whose cells are all bordered by rules becomes one `table` block: one
line per row, the cells separated by ` | `, as in HTML. A table continued on
the next page, with no text or image between the two parts, is joined to its
start, and the header row that it repeats there is removed and counted as
`table_header`. The rows of a table are never taken for running headers or
footers, even close to the edge of the page, and the text before a table is
never joined to the text after it.

The option changes what the audit and the comparison report on PDFs:

- `structure.table_caption` (info) applies to them;
- a `??` in a cell of a table is no longer reported;
- a table read as a table in the reference only gives
  `structure.table_lost`, for information, as every change of kind between
  two PDFs, when the other PDF reads its text in the same order, row by row,
  apart from the text around it, in 100 blocks at most: a split or a merge
  is looked for over 100 blocks, no more. A reportlab table of up to 99 rows
  with rules in one PDF and without them in the other gives it, with
  `block.split_merged`: the other PDF reads its header and each row as a
  paragraph, and the header that it repeats on each page as words added. A
  longer table, or a table that the other PDF reads otherwise, such as a
  booktabs table of pdflatex, which PDFium reads column by column, is
  reported missing, a critical finding, in whole or in part, and its text
  added.

These tables stay text:

- tables without rules, such as booktabs tables, tables with merged cells,
  and tables of fewer than 2 rows or 2 columns;
- tables on a turned page: a page that its `/Rotate` entry turns, or a
  page whose text runs up or down, which the reader turns upright, as for
  a table set sideways;
- tables over a vector figure, such as the help lines of a plot;
- tables in the top or bottom band of a page, such as the ruled header of a
  page template, which stay page chrome;
- tables in a font without Unicode mapping, which pdfminer reads as
  `(cid:N)`: the text of PDFium, where encoding damage stays visible, is
  kept;
- tables whose text, read cell after cell and row after row, differs by a
  single character from the text that PDFium reads in their box, and tables
  that a line of text crosses, such as a word set past the edge of the
  table: a warning counts them.

A table thus takes the place of the text of PDFium only when it holds the
same characters, in the same order, blanks aside, once both are normalized as
the comparison normalizes text (ligatures, quotes, dashes); its cells then
take the blanks of PDFium, so that no word is split or joined to another.
The one exception is a header row that PDFium leaves out, as LibreOffice
marks the header rows that it repeats: that row is left out of the table
too. The text around a table is read in the same order as without the
option.

Pages of more than 20,000 characters, or of too many rules, are not
searched, and a warning counts them.

`--pdf-tables` reads each page a second time, with pdfminer, in a separate
process with limits. The time that the command waits for it, by the clock,
is limited on every system. Its CPU time is limited on Linux and macOS;
Windows has only the clock, which allows three times that CPU time. Its
memory, which it may add 2 GiB of at most, is limited on Linux only. Real
documents take about 40 ms more per page, and 0.3 s
more to start that process. The 1,323 pages of the PGF manual take 91 s
instead of 35 s, and the process holds 202 MB at most beside the 172 MB of
the command: 374 MB together. On a crafted PDF, the reading may spend a few
seconds per page before it stops, with a warning; the document is then read
without its tables.

### Performance

These figures were measured on a laptop with an Intel Core Ultra 7 265U
processor (12 cores, 14 threads) and 64 GB of memory, on AC power, with
Python 3.14: the stages in one process with `tools/benchmark.py`, as
[CONTRIBUTING.md](CONTRIBUTING.md) describes it, and `compare` as a whole
with the `doc-checker` command. The PGF manual is the `pgfmanual.pdf` of
TeX Live, compared with a copy that Ghostscript wrote again. The git
manual is the HTML of `/usr/share/doc/git`, also read four times over in
one site, and compared with a copy where one paragraph in ten is removed
and one in seven has a typo.

| Run | Time | Peak memory |
|---|---|---|
| `extract` of the PGF manual (PDF, 1,323 pages) | 35 s | 170 MB |
| `audit` of the PGF manual | 41 s: extraction 37 s, text 3.6 s | 380 MB |
| `compare` of the PGF manual and its copy | 43 s; in one process 77 s: extraction 35 s and 36 s, comparison 2.9 s, reports 1.3 s | 470 MB; in one process 340 MB |
| `audit` of the git manual (HTML, 259 pages) | 8.1 s: extraction 2.5 s, text 5.2 s | 280 MB |
| `audit` of the git manual four times (1,036 pages) | 28 s: extraction 10 s, text 17 s | 600 MB |
| `compare` of the git manual and its copy | 4.9 s; in one process 6.0 s: extraction 2.0 s and 1.9 s, comparison 1.5 s, reports 0.3 s | 520 MB; in one process 220 MB |

`compare` reads its two inputs in two processes when each one has 1 MB or
more: a PDF file or a single HTML page by its size, a site by the sum of its
`.html` and `.htm` pages. The two processes give the same reports as one,
about 1.8 times faster on the PGF manual, but they hold more memory
together, as the table shows. Smaller inputs are read one after the other in
the process of the command, and so are large ones where the processes cannot
start.

The alignment of the blocks of two documents stops at a work limit, which
grows with their size. Two very different or highly repetitive documents
can reach it. Past it, the blocks left are compared by identical text only,
and a warning of the new document counts them, as when the site that holds
the git manual four times is compared with the same site edited: "47920
block(s) were compared by identical text only: the alignment reached its
work limit". A block moved and edited there is then reported as missing and
added. The real pairs of builds measured use less than a tenth of the limit.

Tables and code blocks of 10,000 characters or more are compared line by
line, in a time that grows with their size, not with its square. Two such
blocks are matched by the lines and the words that they share; their lines
found once in each are matched first, and each run of changed lines is then
compared word by word, but for the words that it starts and ends with on
both sides. A run too long for that, such as a column changed in a thousand
consecutive rows, is reported as one `text.modified`. The largest tables and
code blocks of the git manual and of the PGF manual hold fewer than 7,000
characters. A ruled table of 8,000 rows over 217 pages, a third of its
counts changed, is compared in 14 s with `--pdf-tables`, one finding per
changed row. The word by word comparison of a run takes up to 3 s on crafted
text, such as code blocks of alternating digits: two such blocks of 16 KB
were compared in 25 s, and below 10,000 characters two of 4 KB in 7 s.

### Configuration

Options can also be set in `doc-checker.toml`, or in the
`[tool.doc-checker]` table of `pyproject.toml`:

```toml
[general]
lang = "en"
min_severity = "minor"
output = "doc-checker-report"
formats = ["html", "md"]
images = true        # extract and compare the images

[html]
profile = "auto"     # or the name of a profile, such as "sphinx"
content_selector = "main article"
strip_selectors = [".last-updated"]
include = ["**/*.html", "**/*.htm"]
exclude = ["**/genindex.html", "**/search.html", "drafts/**"]
page_order = "auto"  # or "path"

[pdf]
header_band = 0.08   # top share of each page searched for running headers
footer_band = 0.08   # bottom share searched for running footers
repeat_share = 0.3   # minimum share of the pages a header or footer is on
compare_toc = false  # also compare tables of contents and indexes
tables = false       # read ruled tables as tables (pdf-tables extra)

[ignore]
patterns = ['Last updated on \S+']
rules = ["block.moved", "link.*"]
findings = ["be1135e70f67"]

[severity]
"block.added" = "info"
"structure.kind_changed" = "minor"
```

### Exit codes

| Code | Meaning |
|---|---|
| 0 | The run succeeded, with or without findings |
| 2 | Invalid option or configuration |
| 3 | Unreadable or unsupported input |

## License

doc-checker is released under the Apache License 2.0 (see [LICENSE](LICENSE)).
The French dictionary bundled in the package, from Grammalecte, is released
under the Mozilla Public License 2.0. The licences of the dependencies are in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Contribution

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.
