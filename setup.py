from setuptools import setup

with open('requirements.txt') as f:
    required = f.read().splitlines()  
      
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='meta_tools',
    version='1.0',
    author='Ellariel',
    author_email='d.v.valko@gmail.com',
    description="Some tools for meta-analysis and reporting of research results according to APA standards.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/ellariel/meta_tools',
    license='MIT',
    include_package_data=True,
    package_data={'meta_tools': ['meta_tools/*.py']},
    packages=['meta_tools'],
    install_requires=required,
)
