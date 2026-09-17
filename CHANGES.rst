0.3.11 (unreleased)
-------------------
- Test CASA on Python 3.13; build distributions with Python 3.13 (#156)

0.3.10 (2026-09-03)
-------------------
- Update minimum package dependencies in docs (#138)
- Add `Beam.pixels_per_beam` and `Beams.pixels_per_beam` for converting between beam area and pixel area given a WCS (#150)
- Fix ``Beams.__getitem__`` raising ``ValueError`` on length-1 tuple keys (#152)
- Switch PyPI publishing to trusted publishing (OIDC) (#154)

0.3.9 (2025-06-10)
------------------
- Add ``kwargs`` support to `~radio_beam.Beam.ellipse_to_plot` (#123)
- Escape LaTeX special characters in error/test strings, for Python 3.12+ compatibility (#132)
- CI and packaging maintenance (#136)

0.3.8 (2024-10-18)
------------------
- Fix a `~radio_beam.Beams` regression triggered by astropy issue #17096 (#128)
- Fix ``distutils`` deprecation under Python 3.12 (#124)
- Remove unused ``six``/Python 2 compatibility code (#125)
- CI and dependency maintenance (#126)

0.3.7 (2023-12-07)
------------------
- Update rtd build file #121
- bugfix: brightness_temperature call spec #119
- pdates to testing and such #117

0.3.5/0.3.6 (2023-10-17)
------------------
- Add more documentation about common beam convolution by @keflavich in #113
- remove unused imports that are apparently deprecated anyway by @keflavich in #115
- Clean up _astropy_init.py by @pllim in #116

0.3.4 (2022-09-23)
------------------
 - Simplified two-beam common beam determination (https://github.com/radio-astro-tools/radio-beam/pull/102)
 - Allow non-arcsecond units in beam tables (https://github.com/radio-astro-tools/radio-beam/pull/98)
 - Support for beam initialization from area (https://github.com/radio-astro-tools/radio-beam/pull/94)
 - Drop support for python3.6

0.3.3 (2021-03-19)
------------------
 - Optimized the deconvolution operation to avoid extra unit conversions and creation of new `Beam` objects. (https://github.com/radio-astro-tools/radio-beam/pull/87)

 0.3.2 (2019-08-27)
 ------------------


0.3.1 (2019-02-20)
------------------
 - Set mult/div for convolution/deconvolution in `Beam` and `Beams`.
   The `==` and `!=` operators also work with `Beams` now.
   (https://github.com/radio-astro-tools/radio-beam/pull/75)
 - Added common beam operations to `Beams`.
   (https://github.com/radio-astro-tools/radio-beam/pull/67)
 - Fix PA usage for plotting and kernel routines.
   (https://github.com/radio-astro-tools/radio-beam/pull/65)

0.2 (2017-10-25)
----------------
 - Changed repo name to `radio-beam` from `radio_beam`.
   (https://github.com/radio-astro-tools/radio-beam/pull/59)
 - Enhancement: Added support for multiple beams through the `Beams` class.
   (https://github.com/radio-astro-tools/radio-beam/pull/51)


0.1 (2017-09-08)
----------------
First release
