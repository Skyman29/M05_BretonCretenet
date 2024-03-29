.. image:: https://coveralls.io/repos/github/Skyman29/M05_BretonCretenet/badge.svg?branch=main
   :target: https://coveralls.io/github/Skyman29/M05_BretonCretenet?branch=main
.. image:: https://img.shields.io/badge/docs-latest-orange.svg
   :target: https://skyman29.github.io/M05_BretonCretenet/
.. image:: https://img.shields.io/badge/github-project-0000c0.svg
   :target: https://github.com/Skyman29/M05_BretonCretenet
.. image:: https://img.shields.io/badge/pypi-project-blueviolet.svg
   :target: https://test.pypi.org/project/breton-cretenet/

====================================
Mini project of reproducible science
====================================
|
| This project is a simple toy project. Its only purpose is to explore
  the most important concepts of reproducible science and apply them.
|
| To install the package, type :
| ``pip install .``
|
| You can also download it directly from test Pypi with the following command :
| ``pip install --extra-index-url https://test.pypi.org/simple breton_cretenet``
|
| To run the main program, you can type :
| ``breton_cretenet_results``
|
| To find all the optionnal arguments and possibilities, type :
| ``breton_cretenet_results --help``
|
| For more information, see the documentation via the relevent badge or :
| https://skyman29.github.io/M05_BretonCretenet/
|
| You can find information about the coverage on the relevent badge and if you want to check it yourself, you can run :
| ``pytest -sv --cov-report=term-missing -m "not pull_request" --cov=breton_cretenet breton_cretenet/test.py`` if you want the light version of the tests.
| ``pytest -sv --cov-report=term-missing --cov=breton_cretenet breton_cretenet/test.py`` if you want all the possible combinatins to be tested.
