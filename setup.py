import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
   
#with open("requirements.txt") as f:
#    requirements = f.read().splitlines()

setuptools.setup(name='bregmanet',
      version='1.0.0',
      author='Jordan Frecon',
      author_email='jordan.frecon@gmail.com',
      description='Bregman Neural Networks',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/JordanFrecon/bregmanet',
      license='MIT',
      package_dir={"": "src"},
      packages=setuptools.find_packages(where="src"),
      python_requires=">=3.6",
      #install_requires=requirements,
      )

