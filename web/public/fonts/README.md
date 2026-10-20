# The fonts

Two typefaces, both subsets of Latin, Greek, punctuation and currency, each under its own
licence. The files are the ones the workspace's `tokenlab` ships, copied byte for byte; its
`tools/subset-fonts.sh` made them and records the ranges. Half of this corpus is Greek, and
`tests/test_page.py` holds both files to every character the narrative and the golden
questions use, so none of it falls back to another face.

## Manrope

- **Copyright 2019 The Manrope Project Authors**, <https://github.com/sharanda/manrope>
- **SIL Open Font License 1.1**, full text in [`Manrope-OFL.txt`](Manrope-OFL.txt)
- No Reserved Font Name is declared, so this subset may keep the name Manrope.

## Roboto Mono

- **Copyright 2015 The Roboto Mono Project Authors**, <https://github.com/googlefonts/robotomono>
- **Apache License 2.0**, full text in [`RobotoMono-LICENSE.txt`](RobotoMono-LICENSE.txt)
- Version 3.000, as Google Fonts distributes it, under Apache 2.0 rather than the OFL.

Both binaries keep their copyright and licence records in the name table, name IDs 0 and 13.
