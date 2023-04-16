from setuptools import setup

with open('requirements.txt') as f:
    required = f.read().splitlines()  
      
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='meta_utils',
    version='1.0',
    author='Ellariel',
    author_email='d.v.valko@gmail.com',
    description="Some utils for meta-analysis and reporting with APA standards.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/ellariel/meta_utils',
    license='MIT',
    include_package_data=True,
    package_data={'meta_utils': ['meta_utils/*.py']},
    packages=['meta_utils'],
    install_requires=required,
)
