from setuptools import find_packages, setup


def load_requirements(f):
    retval = [str(k.strip()) for k in open(f, "rt")]
    return [k for k in retval if k and k[0] not in ("#", "-")]


setup(
    name="breton_cretenet",
    version="2.0.0",
    description="Mini-project for the M05 course.",
    url="https://github.com/Cretenet/tests-m05/tree/dev",
    license="MIT",
    author="Quentin Cretenet et Gaetan Breton",
    author_email="qcretenet@hotmail.com",
    long_description=open("README.rst").read(),
    long_description_content_type="text/x-rst",
    packages=find_packages(),
    include_package_data=True,
    install_requires=load_requirements("requirements.txt"),
    entry_points={
        "console_scripts": ["breton_cretenet_results = breton_cretenet.main:main"]
    },
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
